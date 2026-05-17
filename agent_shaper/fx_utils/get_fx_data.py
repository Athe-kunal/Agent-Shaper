"""
get_fx_data.py — per-module and per-function tensor shape metadata for workspace code.

Uses torch.export + ShapeProp to capture every intermediate tensor transformation.
Standalone workspace functions are detected by parsing call-site source lines, because
torch.export inlines them and only records the call site in each node's stack_trace.
"""

from __future__ import annotations

import ast
import inspect
import os
import re
import sys
from collections import defaultdict
from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
from torch.fx.passes.shape_prop import ShapeProp


class TensorInfo(NamedTuple):
    name: str
    shape: Optional[Tuple[int, ...]]
    annotated_shape: Optional[str]   # "(B, T, n_embd)" when dim_names supplied, else None
    dtype: Optional[torch.dtype]
    source_file: Optional[str]
    line_number: Optional[int]


class ModuleInfo(NamedTuple):
    class_name: str
    module_origin: str
    source_file: Optional[str]
    line_start: Optional[int]
    line_end: Optional[int]
    parameters: list   # list[TensorInfo] — nn.Parameters from __init__
    tensors: list      # list[TensorInfo] — every intermediate FX node in forward


class FunctionInfo(NamedTuple):
    func_name: str
    source_file: Optional[str]
    line_start: Optional[int]
    line_end: Optional[int]
    tensors: list      # list[TensorInfo] — FX nodes attributed to this function


class ShapeResult(NamedTuple):
    modules: list    # list[ModuleInfo]
    functions: list  # list[FunctionInfo]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _is_workspace_cls(cls: type, workspace: str) -> bool:
    """Return True if cls is defined inside workspace and not inside site-packages."""
    try:
        f = inspect.getfile(cls)
        return f.startswith(workspace) and "site-packages" not in f
    except (OSError, TypeError):
        return False


def _nearest_workspace_ancestor(qual_path: str, path_to_module: dict, workspace: str):
    """Walk up qual_path to find the nearest ancestor that IS a workspace class.

    Used when a leaf entry in nn_module_stack is a non-workspace module (e.g.
    nn.Linear) whose workspace parent was omitted because it was called via
    .forward() rather than __call__, bypassing torch.export's stack tracking.
    """
    parts = qual_path.split(".")
    for n in range(len(parts) - 1, 0, -1):
        parent_path = ".".join(parts[:n])
        parent_mod = path_to_module.get(parent_path)
        if parent_mod is not None and _is_workspace_cls(type(parent_mod), workspace):
            return parent_path, type(parent_mod)
    return None, None


def _innermost_workspace_origin(node, workspace, path_to_module):
    """Return (qualified_path, cls) for the workspace module owning this node, or (None, None)."""
    stack = node.meta.get("nn_module_stack") or {}
    for key, val in reversed(list(stack.items())):
        if not (isinstance(val, tuple) and len(val) == 2):
            continue
        qual_path, cls_or_str = val
        if isinstance(qual_path, str) and isinstance(cls_or_str, str):
            mod = path_to_module.get(qual_path)
            if mod is not None:
                cls = type(mod)
                if _is_workspace_cls(cls, workspace):
                    return qual_path, cls
                ancestor_path, ancestor_cls = _nearest_workspace_ancestor(
                    qual_path, path_to_module, workspace
                )
                if ancestor_path is not None:
                    return ancestor_path, ancestor_cls
        elif isinstance(cls_or_str, type) and _is_workspace_cls(cls_or_str, workspace):
            return qual_path, cls_or_str
    return None, None


def _node_callsites(node, workspace: str) -> list[tuple[str, int]]:
    """Return all (abs_file, line_no) workspace frames from stack_trace, innermost first.

    torch.export inlines standalone functions, so for a call chain like
    exec_script → outer_fn (opd.py) → inner_fn (opd.py) → primitive_op, the
    stack trace carries multiple workspace frames. Returning all of them lets
    Pass 2 walk outward until it finds a frame whose source line names a
    registry function — handling callers that are not themselves workspace files.
    """
    trace = node.meta.get("stack_trace", "")
    if not trace:
        return []
    result = []
    for f, line in reversed(re.findall(r'File "([^"]+)", line (\d+)', trace)):
        abs_f = os.path.abspath(f)
        if abs_f.startswith(workspace) and "site-packages" not in abs_f:
            result.append((abs_f, int(line)))
    return result


def _build_workspace_func_registry(workspace: str) -> dict[str, list[tuple]]:
    """Return {func_name: [(abs_file, rel_file, line_start, line_end)]} for all top-level workspace functions."""
    registry: dict[str, list[tuple]] = defaultdict(list)
    for root, dirs, files in os.walk(workspace):
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".") and d not in ("__pycache__", "venv")
            and "site-packages" not in os.path.join(root, d)
        ]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            abs_f = os.path.join(root, fname)
            if "site-packages" in abs_f:
                continue
            try:
                with open(abs_f) as fh:
                    src = fh.read()
                tree = ast.parse(src)
                rel_f = os.path.relpath(abs_f)
                for child in ast.iter_child_nodes(tree):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        registry[child.name].append((abs_f, rel_f, child.lineno, child.end_lineno))
            except (OSError, SyntaxError):
                continue
    return dict(registry)


def _free_func_names_at_line(src_lines: list[str], line_no: int) -> list[str]:
    """Return names of all free-function calls (not method calls) on the given source line."""
    if line_no <= 0 or line_no > len(src_lines):
        return []
    line = src_lines[line_no - 1].strip()
    try:
        tree = ast.parse(line, mode="eval")
    except SyntaxError:
        try:
            tree = ast.parse(line, mode="exec")
        except SyntaxError:
            return []
    return [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]


def _node_shape(node):
    """Return (shape, dtype) from node.meta['tensor_meta'], handling single and tuple outputs."""
    meta = node.meta.get("tensor_meta")
    if meta is None:
        return None, None
    if hasattr(meta, "shape"):
        return tuple(meta.shape), meta.dtype
    shapes = [tuple(m.shape) if m is not None else None for m in meta]
    dtypes = [m.dtype if m is not None else None for m in meta]
    return shapes, dtypes


def _node_line(node, abs_src_file):
    """Return the source line number in abs_src_file where this node was created, or None."""
    trace = node.meta.get("stack_trace", "")
    if not trace or not abs_src_file:
        return None
    for f, line in reversed(re.findall(r'File "([^"]+)", line (\d+)', trace)):
        if os.path.abspath(f) == abs_src_file:
            return int(line)
    return None


def _param_line(src_lines, class_start, param_name):
    """Return the absolute line number of the first 'self.<param_name> = ...' assignment in src_lines."""
    pattern = re.compile(rf"\bself\.{re.escape(param_name)}\s*=")
    for i, line in enumerate(src_lines):
        if pattern.search(line):
            return class_start + i
    return None


def _cls_meta(cls):
    """Return (src_lines, line_start, line_end, rel_src_file, abs_src_file) for a class."""
    try:
        src_lines, start = inspect.getsourcelines(cls)
        src_file = os.path.relpath(inspect.getfile(cls))
        abs_file = os.path.abspath(src_file)
    except (OSError, TypeError):
        src_lines, start, src_file, abs_file = [], None, None, None
    end = (start + len(src_lines) - 1) if start is not None else None
    return src_lines, start, end, src_file, abs_file


def _node_op_name(node) -> str:
    """Return a human-readable operation name for an FX node (method name, function name, or module leaf name)."""
    if node.op == "call_method":
        return str(node.target)
    if node.op == "call_function":
        return getattr(node.target, "__name__", str(node.target))
    if node.op == "call_module":
        return str(node.target).split(".")[-1]
    return node.op


def _cls_varnames(cls, class_start: Optional[int]) -> dict[int, str]:
    """Return {abs_line: var_name} for every assignment in every method of cls."""
    if class_start is None:
        return {}
    try:
        src = inspect.getsource(cls)
        tree = ast.parse(src)
    except (OSError, TypeError, SyntaxError):
        return {}

    offset = class_start - 1
    result: dict[int, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Assign) and stmt.targets:
                target = stmt.targets[0]
            elif isinstance(stmt, ast.AugAssign):
                target = stmt.target
            else:
                continue
            abs_line = stmt.lineno + offset
            if isinstance(target, ast.Name):
                result[abs_line] = target.id
            elif isinstance(target, ast.Tuple):
                names = [e.id for e in target.elts if isinstance(e, ast.Name)]
                if names:
                    result[abs_line] = f"({', '.join(names)})"
    return result



def _parse_dim_bounds(error_msg: str) -> dict[str, dict[str, int]]:
    """Extract per-Dim bounds from a ConstraintViolationError's 'Suggested fixes' section.

    Parses patterns like Dim('T', max=32) into {"T": {"max": 31}}.
    Subtracts 1 from max because PyTorch's suggestion reflects the embedding table size
    but the model guard is strict (t < block_size), so the true upper bound is max - 1.
    """
    bounds: dict[str, dict[str, int]] = {}
    for m in re.finditer(r"Dim\('(\w+)'([^)]*)\)", error_msg):
        name, kwargs_str = m.group(1), m.group(2)
        entry = {}
        for kw, val in re.findall(r'(min|max)=(\d+)', kwargs_str):
            entry[kw] = int(val) - 1 if kw == "max" else int(val)
        if entry:
            bounds[name] = entry
    return bounds


def _parse_ne_guard_dim_names(error_msg: str) -> set[str]:
    """Extract dim names that appear in != guards from offset slicing.

    e.g. logits[:, :-1, :] produces:
      Not all values of S ... satisfy the guard ((-1) + ...) != 1.

    torch.export's constraint solver cannot discharge these guards even with a
    declared min bound, so the affected dims must be made static on retry.
    """
    names: set[str] = set()
    for m in re.finditer(
        r"Not all values of (\w+) = L\[.*?\]\.size\(\)\[(\d+)\].*?"
        r"\(\((-?\d+)\) \+ L\[.*?\]\.size\(\)\[\d+\]\) != \d+",
        error_msg,
        re.DOTALL,
    ):
        names.add(m.group(1))
    return names


def _build_dynamic_shapes(
    module: nn.Module,
    example_args: tuple,
    dim_names: dict[str, int],
    dim_bounds: Optional[dict[str, dict[str, int]]] = None,
) -> dict[str, dict[int, object]]:
    """Convert {name: value} dim_names into a dynamic_shapes dict for torch.export.export.

    Raises ValueError on collision (two names with the same concrete value).
    Axis-matches by value: every axis in every example_arg tensor whose size equals
    a known value gets mapped to the corresponding torch.export.Dim symbol.
    dim_bounds, when provided, sets min/max constraints on each Dim (used on retry
    after torch.export raises ConstraintViolationError with suggested fixes).
    """
    val_to_name: dict[int, str] = {}
    for name, val in dim_names.items():
        if val in val_to_name:
            raise ValueError(
                f"dim_names collision: '{name}' and '{val_to_name[val]}' both have value {val}. "
                "Use distinct example sizes."
            )
        val_to_name[val] = name

    dims = {
        name: torch.export.Dim(name, **((dim_bounds or {}).get(name, {})))
        for name in dim_names
    }

    sig = inspect.signature(module.forward)
    param_names = list(sig.parameters.keys())

    dynamic_shapes: dict[str, dict[int, object]] = {}
    for param_name, arg in zip(param_names, example_args):
        if not isinstance(arg, torch.Tensor):
            continue
        axis_map = {
            axis: dims[val_to_name[size]]
            for axis, size in enumerate(arg.shape)
            if size in val_to_name
        }
        if axis_map:
            dynamic_shapes[param_name] = axis_map

    return dynamic_shapes


def _node_annotated_shape(
    node,
    dynamic_val_to_name: dict[int, str],
    static_val_to_name: dict[int, str],
) -> Optional[str]:
    """Return a symbolic shape string by reading node.meta['val'] (FakeTensor from torch.export).

    Dynamic axes (SymInt) are resolved via dynamic_val_to_name keyed by the hint value.
    Static axes (plain int) are resolved via static_val_to_name — a separate dict built
    from module attributes, excluding dynamic dim values to prevent mis-labelling
    (e.g. n_head=2 shown as 'B' when B=2).
    Returns None if val metadata is absent.
    """
    def _fmt_dim(d) -> str:
        if isinstance(d, torch.SymInt):
            return dynamic_val_to_name.get(d.node.hint, str(d))
        # Exact match: declared dynamic dim forced static, or module attribute
        if d in dynamic_val_to_name:
            return dynamic_val_to_name[d]
        if d in static_val_to_name:
            return static_val_to_name[d]
        # Derived: d = declared_dim ± small_offset (e.g. S-1 from [:, :-1, :])
        for val, name in dynamic_val_to_name.items():
            diff = d - val
            if 0 < abs(diff) <= 3:
                return f"{name}{'+' if diff > 0 else '-'}{abs(diff)}"
        return str(d)

    def _fmt_shape(shape) -> str:
        return f"({', '.join(_fmt_dim(d) for d in shape)})"

    val = node.meta.get("val")
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        parts = [_fmt_shape(v.shape) if hasattr(v, "shape") else "?" for v in val]
        return f"[{', '.join(parts)}]"
    if not hasattr(val, "shape"):
        return None
    return _fmt_shape(val.shape)


def _hashable_shape(shape):
    """Convert a shape (possibly a list of shapes for tuple outputs) to a hashable form for deduplication."""
    if shape is None:
        return None
    if isinstance(shape, list):
        return tuple(tuple(s) if s is not None else None for s in shape)
    return shape


# ─── Module attribute dim inference ──────────────────────────────────────────

def _collect_module_attr_names(
    path_to_module: dict,
    workspace: str,
    dynamic_values: set[int],
) -> dict[int, str]:
    """Return {int_value: attr_name} from integer attributes of all workspace submodules.

    Skips values already used by dynamic dims (user-supplied via dim_names) so that
    a static attribute whose value happens to equal a dynamic dim (e.g. n_head=2 when B=2)
    is never confused with that dynamic dim in annotations.
    Only collects positive integers > 1 to avoid noise from flags and indices.
    First attribute name seen wins when multiple attrs share the same value.
    """
    result: dict[int, str] = {}

    for sub_mod in path_to_module.values():
        if not _is_workspace_cls(type(sub_mod), workspace):
            continue
        for attr, val in vars(sub_mod).items():
            if not isinstance(val, int) or val <= 1 or attr.startswith("_"):
                continue
            if val in dynamic_values or val in result:
                continue
            result[val] = attr

    return result


# ─── Export cache ─────────────────────────────────────────────────────────────

_export_cache: dict = {}


def _export_cache_key(module: nn.Module, example_args: tuple) -> tuple:
    """Build a hashable cache key from module class name and arg shapes/dtypes."""
    arg_sig = tuple(
        (tuple(a.shape), a.dtype) if isinstance(a, torch.Tensor) else a
        for a in example_args
    )
    return (type(module).__name__, arg_sig)


# ─── Eager line tracer for standalone functions ───────────────────────────────

def _fmt_dim_eager(d: int, dynamic_val_to_name: dict, static_val_to_name: dict) -> str:
    """Format a single integer dimension using the same logic as _node_annotated_shape."""
    if d in dynamic_val_to_name:
        return dynamic_val_to_name[d]
    if d in static_val_to_name:
        return static_val_to_name[d]
    for val, name in dynamic_val_to_name.items():
        diff = d - val
        if 0 < abs(diff) <= 3:
            return f"{name}{'+' if diff > 0 else '-'}{abs(diff)}"
    return str(d)


def _make_eager_annotated_shape(
    tensor: torch.Tensor,
    dynamic_val_to_name: dict,
    static_val_to_name: dict,
) -> Optional[str]:
    dims = [_fmt_dim_eager(d, dynamic_val_to_name, static_val_to_name) for d in tensor.shape]
    return f"({', '.join(dims)})"


def _trace_function_lines(
    model: nn.Module,
    example_args: tuple,
    workspace: str,
    dynamic_val_to_name: dict,
    static_val_to_name: dict,
) -> dict[tuple, list]:
    """Run an eager forward pass with sys.settrace to capture per-line tensor shapes.

    Returns {(rel_file, lineno): [TensorInfo]} for every line in workspace standalone
    functions (i.e. non-__init__, non-forward methods and module class files are excluded).

    Strategy: sys.settrace fires 'line' events BEFORE a line executes. At line L, the
    locals reflect the state after all lines < L. So we diff current locals vs. the
    previous snapshot to find tensors assigned by line L-1, and attribute them to L-1.
    """
    result: defaultdict = defaultdict(list)
    seen: set = set()

    # Per-frame state: {frame_id: (prev_lineno, prev_locals_snapshot)}
    frame_state: dict = {}

    def _record_new_tensors(rel_file: str, lineno: int, new_vars: dict) -> None:
        for var_name, val in new_vars.items():
            if not isinstance(val, torch.Tensor):
                continue
            if var_name.startswith("_"):
                continue
            shape = tuple(val.shape)
            dedup = (rel_file, lineno, var_name, shape)
            if dedup in seen:
                continue
            seen.add(dedup)
            result[(rel_file, lineno)].append(TensorInfo(
                name=var_name,
                shape=shape,
                annotated_shape=_make_eager_annotated_shape(val, dynamic_val_to_name, static_val_to_name),
                dtype=val.dtype,
                source_file=rel_file,
                line_number=lineno,
            ))

    def tracer(frame, event, arg):
        if event not in ("line", "return"):
            return tracer
        filename = frame.f_code.co_filename
        abs_file = os.path.abspath(filename)
        if not abs_file.startswith(workspace) or "site-packages" in abs_file:
            return tracer

        rel_file = os.path.relpath(abs_file)
        curr_lineno = frame.f_lineno
        fid = id(frame)

        curr_locals = {
            k: v for k, v in frame.f_locals.items()
            if isinstance(v, torch.Tensor)
        }

        if fid in frame_state:
            prev_lineno, prev_locals = frame_state[fid]
            # Find tensors that are new or changed since prev snapshot.
            new_vars = {
                k: v for k, v in curr_locals.items()
                if k not in prev_locals or prev_locals[k] is not v
            }
            if new_vars:
                _record_new_tensors(rel_file, prev_lineno, new_vars)

        frame_state[fid] = (curr_lineno, curr_locals)
        return tracer

    with torch.no_grad():
        old_trace = sys.gettrace()
        sys.settrace(tracer)
        try:
            model(*example_args)
        finally:
            sys.settrace(old_trace)

    return dict(result)


# ─── Line-map builder ─────────────────────────────────────────────────────────

def _build_line_map_from_result(
    shape_result: "ShapeResult",
) -> dict[tuple, list]:
    """Build {(rel_file, lineno): [TensorInfo]} from a ShapeResult (modules + functions)."""
    result: defaultdict = defaultdict(list)
    seen: set = set()

    all_infos = list(shape_result.modules) + list(shape_result.functions)
    for info in all_infos:
        params = getattr(info, "parameters", [])
        for tensor in params + info.tensors:
            if tensor.source_file is None or tensor.line_number is None:
                continue
            dedup = (
                tensor.source_file,
                tensor.line_number,
                tensor.name,
                _hashable_shape(tensor.shape),
            )
            if dedup in seen:
                continue
            seen.add(dedup)
            result[(tensor.source_file, tensor.line_number)].append(tensor)

    return dict(result)


# ─── Public API ───────────────────────────────────────────────────────────────

def get_module_shapes(
    module: nn.Module,
    example_args: tuple,
    workspace: Optional[str] = None,
    dim_names: Optional[dict[str, int]] = None,
) -> ShapeResult:
    """
    Parameters
    ----------
    dim_names:
        Optional ``{name: value}`` mapping. Converted internally to torch.export.Dim
        symbols so annotation is structural, not value-matched. Raises ValueError on
        collision (two names with the same concrete value).

    Returns
    -------
    ShapeResult
        .modules   — one ModuleInfo per unique workspace nn.Module in the graph
        .functions — one FunctionInfo per unique workspace standalone function in the graph
    """
    workspace = workspace or os.getcwd()
    module = module.eval()
    path_to_module = dict(module.named_modules())

    dynamic_shapes = _build_dynamic_shapes(module, example_args, dim_names)
    dynamic_val_to_name: dict[int, str] = {v: k for k, v in (dim_names or {}).items()}
    static_val_to_name = _collect_module_attr_names(
        path_to_module, workspace, set(dynamic_val_to_name)
    )

    cache_key = _export_cache_key(module, example_args)
    if cache_key not in _export_cache:
        try:
            exported = torch.export.export(module, example_args, dynamic_shapes=dynamic_shapes)
        except torch._dynamo.exc.UserError as e:
            error_str = str(e)
            bounds = _parse_dim_bounds(error_str)
            # Dims in != guards (e.g. S-1 != 1 from [:, :-1, :]) cannot be resolved
            # by tightening bounds — the constraint solver doesn't propagate bounds
            # through arithmetic. Make them static so export succeeds.
            ne_dims = _parse_ne_guard_dim_names(error_str)
            retry_dim_names = {k: v for k, v in (dim_names or {}).items() if k not in ne_dims}
            dynamic_shapes = _build_dynamic_shapes(module, example_args, retry_dim_names, dim_bounds=bounds)
            exported = torch.export.export(module, example_args, dynamic_shapes=dynamic_shapes)
        _export_cache[cache_key] = exported

    gm = _export_cache[cache_key].module()
    ShapeProp(gm).propagate(*example_args)

    # Build a registry of top-level workspace functions once.
    # torch.export inlines functions so their body never appears in stack_trace;
    # instead we detect them by parsing the call-site line.
    func_registry = _build_workspace_func_registry(workspace)
    src_line_cache: dict[str, list[str]] = {}  # abs_file -> source lines

    cls_meta_cache: dict[type, tuple] = {}

    def get_cls_meta(cls):
        """Return cached (src_lines, start, end, src_file, abs_file, varnames) for cls."""
        if cls not in cls_meta_cache:
            src_lines, start, end, src_file, abs_file = _cls_meta(cls)
            varnames = _cls_varnames(cls, start)
            cls_meta_cache[cls] = (src_lines, start, end, src_file, abs_file, varnames)
        return cls_meta_cache[cls]

    def get_src_lines(abs_file: str) -> list[str]:
        """Return cached source lines for abs_file, reading from disk on first access."""
        if abs_file not in src_line_cache:
            try:
                with open(abs_file) as fh:
                    src_line_cache[abs_file] = fh.readlines()
            except OSError:
                src_line_cache[abs_file] = []
        return src_line_cache[abs_file]

    # ── Pass 1: group FX nodes by workspace nn.Module ──────────────────────────
    mod_seen_keys: list[tuple] = []
    mod_key_to_tensors: dict[tuple, list[TensorInfo]] = {}

    for node in gm.graph.nodes:
        origin, cls = _innermost_workspace_origin(node, workspace, path_to_module)
        if cls is None:
            continue
        key = (origin, cls)
        shape, dtype = _node_shape(node)
        src_lines, start, end, src_file, abs_file, varnames = get_cls_meta(cls)
        line_no = _node_line(node, abs_file)
        var = varnames.get(line_no) if line_no is not None else None
        tensor_name = f"{var} [{_node_op_name(node)}]" if var is not None else node.name
        t = TensorInfo(
            name=tensor_name,
            shape=shape,
            annotated_shape=_node_annotated_shape(node, dynamic_val_to_name, static_val_to_name),
            dtype=dtype,
            source_file=src_file,
            line_number=line_no,
        )
        if key not in mod_key_to_tensors:
            mod_seen_keys.append(key)
            mod_key_to_tensors[key] = []
        mod_key_to_tensors[key].append(t)

    modules: list[ModuleInfo] = []
    mod_seen_dedup: set[tuple] = set()

    for (origin, cls) in mod_seen_keys:
        tensors = mod_key_to_tensors[(origin, cls)]
        dedup_key = (cls.__name__, tuple(_hashable_shape(t.shape) for t in tensors))
        if dedup_key in mod_seen_dedup:
            continue
        mod_seen_dedup.add(dedup_key)

        src_lines, start, end, src_file, abs_file, varnames = get_cls_meta(cls)
        mod = path_to_module.get(origin, None)
        params = []
        if mod is not None:
            params = [
                TensorInfo(
                    name=name,
                    shape=tuple(p.shape),
                    annotated_shape=None,
                    dtype=p.dtype,
                    source_file=src_file,
                    line_number=_param_line(src_lines, start, name) if start is not None else None,
                )
                for name, p in mod.named_parameters(recurse=False)
            ]

        modules.append(ModuleInfo(
            class_name=cls.__name__,
            module_origin=origin or "root",
            source_file=src_file,
            line_start=start,
            line_end=end,
            parameters=params,
            tensors=tensors,
        ))

    # ── Pass 2: attribute FX nodes to workspace standalone functions ────────────
    # torch.export inlines standalone functions, so stack_trace only records the
    # call site in forward(). We parse that line to find which workspace function
    # is being called, then group nodes by function.
    func_seen_keys: list[tuple] = []
    func_key_to_tensors: dict[tuple, list[TensorInfo]] = {}
    func_key_meta: dict[tuple, tuple] = {}  # (abs_file, func_name) -> (rel_file, line_start, line_end)

    for node in gm.graph.nodes:
        # Walk workspace frames from innermost outward; stop at the first frame
        # whose source line names a registry function.  This handles both the
        # standard case (workspace forward() calls a workspace function — innermost
        # frame is the call site) and the exec-script wrapper case (non-workspace
        # forward() calls workspace functions — innermost frames are inside function
        # bodies; the relevant call-site frame is one or more levels up).
        for call_abs_file, call_line_no in _node_callsites(node, workspace):
            src_lines = get_src_lines(call_abs_file)
            called_names = _free_func_names_at_line(src_lines, call_line_no)

            matched = False
            for func_name in called_names:
                if func_name not in func_registry:
                    continue
                matched = True
                for abs_func_file, rel_func_file, line_start, line_end in func_registry[func_name]:
                    key = (abs_func_file, func_name)
                    if key not in func_key_meta:
                        func_key_meta[key] = (rel_func_file, line_start, line_end)

                    # Find the body line: innermost frame inside the function's file.
                    # torch.export preserves the full call stack, so the innermost
                    # workspace frame for an inlined function is the exact body line.
                    body_line = next(
                        (ln for fa, ln in _node_callsites(node, workspace) if fa == abs_func_file),
                        None,
                    )

                    shape, dtype = _node_shape(node)
                    t = TensorInfo(
                        name=node.name,
                        shape=shape,
                        annotated_shape=_node_annotated_shape(node, dynamic_val_to_name, static_val_to_name),
                        dtype=dtype,
                        source_file=rel_func_file,
                        line_number=body_line,
                    )
                    if key not in func_key_to_tensors:
                        func_seen_keys.append(key)
                        func_key_to_tensors[key] = []
                    func_key_to_tensors[key].append(t)

            if matched:
                break  # innermost registry match wins; don't attribute to outer callers

    functions: list[FunctionInfo] = []
    func_seen_dedup: set[tuple] = set()

    for (abs_func_file, func_name) in func_seen_keys:
        tensors = func_key_to_tensors[(abs_func_file, func_name)]
        dedup_key = (func_name, abs_func_file, tuple(_hashable_shape(t.shape) for t in tensors))
        if dedup_key in func_seen_dedup:
            continue
        func_seen_dedup.add(dedup_key)

        rel_file, line_start, line_end = func_key_meta[(abs_func_file, func_name)]
        functions.append(FunctionInfo(
            func_name=func_name,
            source_file=rel_file,
            line_start=line_start,
            line_end=line_end,
            tensors=tensors,
        ))

    return ShapeResult(modules=modules, functions=functions)


def get_annotated_line_map(
    module: nn.Module,
    example_args: tuple,
    workspace: Optional[str] = None,
    dim_names: Optional[dict[str, int]] = None,
) -> dict[tuple, list]:
    """Return a merged {(rel_file, lineno): [TensorInfo]} from FX tracing and eager line tracing.

    FX tracing (torch.export) provides per-module tensor shapes with symbolic dim names.
    Eager line tracing (sys.settrace) fills in per-line shapes for standalone functions
    whose bodies are invisible to torch.export (it inlines them, losing body frames).
    The two sources are merged, deduplicating by (file, lineno, name, shape).

    Parameters
    ----------
    module:
        The nn.Module to trace.
    example_args:
        Tuple of example tensors for the module's forward() signature.
    workspace:
        Root directory to restrict tracing to. Defaults to os.getcwd().
    dim_names:
        Optional {name: value} mapping for symbolic dim annotation.

    Returns
    -------
    dict mapping (rel_file, lineno) -> list[TensorInfo], covering both module
    and standalone-function lines.
    """
    workspace = workspace or os.getcwd()
    shape_result = get_module_shapes(module, example_args, workspace=workspace, dim_names=dim_names)

    # FX-based line map (modules + any FX-visible function nodes)
    line_map = _build_line_map_from_result(shape_result)

    # Eager line tracer for standalone function bodies
    dynamic_val_to_name: dict[int, str] = {v: k for k, v in (dim_names or {}).items()}
    path_to_module = dict(module.named_modules())
    static_val_to_name = _collect_module_attr_names(
        path_to_module, workspace, set(dynamic_val_to_name)
    )
    eager_map = _trace_function_lines(
        module, example_args, workspace, dynamic_val_to_name, static_val_to_name
    )

    # Merge: eager entries supplement FX entries; don't overwrite existing names
    seen_existing: dict[tuple, set] = {}
    for (src_file, lineno), tensors in line_map.items():
        seen_existing[(src_file, lineno)] = {t.name for t in tensors}

    merged = dict(line_map)
    for (src_file, lineno), tensors in eager_map.items():
        existing_names = seen_existing.get((src_file, lineno), set())
        new_tensors = [t for t in tensors if t.name not in existing_names]
        if not new_tensors:
            continue
        if (src_file, lineno) in merged:
            merged[(src_file, lineno)] = merged[(src_file, lineno)] + new_tensors
        else:
            merged[(src_file, lineno)] = new_tensors

    return merged


# ─── CLI smoke-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from agent_shaper.transformer.model import GPT, GPTConfig

    cfg = GPTConfig(
        block_size=32,
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
        bias=True,
    )

    B, T = 2, 16
    example_args = (
        torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.long),
        torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.long),
    )

    dim_names = {"B": B, "T": T}

    result = get_module_shapes(GPT(cfg), example_args, dim_names=dim_names)

    SEP = "═" * 80
    for g in result.modules:
        print(f"\n{SEP}")
        print(f"  {g.class_name}  [{g.module_origin}]  L{g.line_start}–L{g.line_end}  {g.source_file}")
        print(SEP)
        if g.parameters:
            print("  Parameters:")
            for t in g.parameters:
                ann = f"  {t.annotated_shape}" if t.annotated_shape else ""
                print(f"    {t.name:<28}  shape={t.shape}{ann}  {t.source_file}:L{t.line_number}")
        if g.tensors:
            print("  Tensors:")
            for t in g.tensors:
                ann = f"  {t.annotated_shape}" if t.annotated_shape else ""
                print(f"    {t.name:<28}  shape={t.shape}{ann}  {t.source_file}:L{t.line_number}")

    for f in result.functions:
        print(f"\n{SEP}")
        print(f"  fn {f.func_name}  L{f.line_start}–L{f.line_end}  {f.source_file}")
        print(SEP)
        for t in f.tensors:
            ann = f"  {t.annotated_shape}" if t.annotated_shape else ""
            print(f"    {t.name:<28}  shape={t.shape}{ann}  {t.source_file}:L{t.line_number}")
    
    
