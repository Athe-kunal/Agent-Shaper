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
import torch.fx
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
    """Return (qualified_path, cls, via_ancestor) for the workspace module owning this node.

    via_ancestor=True means the match was found by walking up from a non-workspace
    child (e.g. nn.Linear inside SelfAttention called via .forward()), not from a
    direct nn_module_stack entry for that class. Callers use this flag to decide
    whether to supplement with a symbolic-trace pass.

    Returns (None, None, False) when no workspace class can be attributed.
    """
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
                    return qual_path, cls, False
                # Not workspace; its workspace parent may have been skipped
                # because it was called via .forward() instead of __call__.
                ancestor_path, ancestor_cls = _nearest_workspace_ancestor(
                    qual_path, path_to_module, workspace
                )
                if ancestor_path is not None:
                    return ancestor_path, ancestor_cls, True
        elif isinstance(cls_or_str, type) and _is_workspace_cls(cls_or_str, workspace):
            return qual_path, cls_or_str, False
    return None, None, False


def _node_callsite(node, workspace: str):
    """Return (abs_file, line_no) of the innermost workspace frame in stack_trace."""
    trace = node.meta.get("stack_trace", "")
    if not trace:
        return None, None
    for f, line in reversed(re.findall(r'File "([^"]+)", line (\d+)', trace)):
        abs_f = os.path.abspath(f)
        if abs_f.startswith(workspace) and "site-packages" not in abs_f:
            return abs_f, int(line)
    return None, None


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
    meta = node.meta.get("tensor_meta")
    if meta is None:
        return None, None
    if hasattr(meta, "shape"):
        return tuple(meta.shape), meta.dtype
    shapes = [tuple(m.shape) if m is not None else None for m in meta]
    dtypes = [m.dtype if m is not None else None for m in meta]
    return shapes, dtypes


def _node_line(node, abs_src_file):
    trace = node.meta.get("stack_trace", "")
    if not trace or not abs_src_file:
        return None
    for f, line in reversed(re.findall(r'File "([^"]+)", line (\d+)', trace)):
        if os.path.abspath(f) == abs_src_file:
            return int(line)
    return None


def _param_line(src_lines, class_start, param_name):
    pattern = re.compile(rf"\bself\.{re.escape(param_name)}\s*=")
    for i, line in enumerate(src_lines):
        if pattern.search(line):
            return class_start + i
    return None


def _cls_meta(cls):
    try:
        src_lines, start = inspect.getsourcelines(cls)
        src_file = os.path.relpath(inspect.getfile(cls))
        abs_file = os.path.abspath(src_file)
    except (OSError, TypeError):
        src_lines, start, src_file, abs_file = [], None, None, None
    end = (start + len(src_lines) - 1) if start is not None else None
    return src_lines, start, end, src_file, abs_file


def _node_op_name(node) -> str:
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


def _build_val_to_names(dim_names: dict[str, int]) -> dict[int, list[str]]:
    result: dict[int, list[str]] = defaultdict(list)
    for name, val in dim_names.items():
        result[val].append(name)
    return dict(result)


def _annotate(shape, val_to_names: dict[int, list[str]]) -> Optional[str]:
    """Return a symbolic shape string, e.g. '(B, T, n_embd)'."""
    if shape is None or not val_to_names:
        return None
    if isinstance(shape, list):
        return f"[{', '.join(_annotate(s, val_to_names) or '?' for s in shape)}]"
    parts = ["/".join(val_to_names[d]) if d in val_to_names else str(d) for d in shape]
    return f"({', '.join(parts)})"


def _hashable_shape(shape):
    if shape is None:
        return None
    if isinstance(shape, list):
        return tuple(tuple(s) if s is not None else None for s in shape)
    return shape


# ─── Symbolic-trace fallback ──────────────────────────────────────────────────

def _wrap_file_fns(sub_mod: nn.Module) -> None:
    """Register all standalone functions from sub_mod's source file as FX leaf nodes.

    This prevents symbolic_trace from trying to trace into functions like
    apply_rotary_embeddings that iterate over Proxy objects (causing errors).
    """
    try:
        src_file = os.path.abspath(inspect.getfile(type(sub_mod)))
    except (OSError, TypeError):
        return
    for mod in sys.modules.values():
        try:
            mod_file = getattr(mod, "__file__", None)
            if not mod_file or os.path.abspath(mod_file) != src_file:
                continue
            for name in list(vars(mod)):
                obj = getattr(mod, name, None)
                if callable(obj) and not isinstance(obj, (type, nn.Module)):
                    try:
                        torch.fx.wrap(obj)
                    except Exception:
                        pass
        except Exception:
            continue


def _symbolic_trace_fallback(
    sub_mod: nn.Module,
    input_args: tuple,
    workspace: str,
    val_to_names: dict,
    get_cls_meta,
) -> list[TensorInfo]:
    """Attempt torch.fx.symbolic_trace for a module torch.export couldn't handle.

    Splits input_args into tensor args (traced symbolically) and non-tensor args
    (frozen as concrete_args so the tracer can handle integer slice indices, etc.).
    Returns a list of TensorInfo, or [] if tracing or shape propagation fail.
    """
    cls = type(sub_mod)

    try:
        sig = inspect.signature(sub_mod.forward)
        param_names = list(sig.parameters.keys())
    except (ValueError, TypeError):
        return []

    concrete_args: dict = {}
    tensor_args: list = []
    for name, arg in zip(param_names, input_args):
        if isinstance(arg, torch.Tensor):
            tensor_args.append(arg)
        else:
            concrete_args[name] = arg

    traced = None
    for attempt in range(2):
        try:
            traced = torch.fx.symbolic_trace(
                sub_mod,
                concrete_args=concrete_args if concrete_args else None,
            )
            break
        except Exception as exc:
            if attempt == 0 and "cannot be iterated" in str(exc):
                _wrap_file_fns(sub_mod)
                continue
            return []
    if traced is None:
        return []

    try:
        ShapeProp(traced).propagate(*tensor_args)
    except Exception:
        return []

    _, _, _, src_file, abs_file, varnames = get_cls_meta(cls)

    tensors: list[TensorInfo] = []
    for node in traced.graph.nodes:
        if node.op not in ("call_function", "call_method", "call_module"):
            continue
        shape, dtype = _node_shape(node)
        if shape is None:
            continue
        line_no = _node_line(node, abs_file)
        var = varnames.get(line_no) if line_no is not None else None
        tensor_name = f"{var} [{_node_op_name(node)}]" if var is not None else node.name
        tensors.append(TensorInfo(
            name=tensor_name,
            shape=shape,
            annotated_shape=_annotate(shape, val_to_names),
            dtype=dtype,
            source_file=src_file,
            line_number=line_no,
        ))
    return tensors


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
        Optional ``{name: value}`` mapping for symbolic shape annotation.

    Returns
    -------
    ShapeResult
        .modules   — one ModuleInfo per unique workspace nn.Module in the graph
        .functions — one FunctionInfo per unique workspace standalone function in the graph
    """
    workspace = workspace or os.getcwd()
    module = module.eval()
    path_to_module = dict(module.named_modules())
    val_to_names = _build_val_to_names(dim_names) if dim_names else {}

    exported = torch.export.export(module, example_args)
    gm = exported.module()
    ShapeProp(gm).propagate(*example_args)

    # Build a registry of top-level workspace functions once.
    # torch.export inlines functions so their body never appears in stack_trace;
    # instead we detect them by parsing the call-site line.
    func_registry = _build_workspace_func_registry(workspace)
    src_line_cache: dict[str, list[str]] = {}  # abs_file -> source lines

    cls_meta_cache: dict[type, tuple] = {}

    def get_cls_meta(cls):
        if cls not in cls_meta_cache:
            src_lines, start, end, src_file, abs_file = _cls_meta(cls)
            varnames = _cls_varnames(cls, start)
            cls_meta_cache[cls] = (src_lines, start, end, src_file, abs_file, varnames)
        return cls_meta_cache[cls]

    def get_src_lines(abs_file: str) -> list[str]:
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
    # Classes whose ops were attributed only via ancestor-walk (called via
    # .forward()), meaning their annotation is incomplete — only child-module
    # ops are captured, not their own intermediate tensor ops.
    ancestor_only_classes: set[str] = set()

    for node in gm.graph.nodes:
        origin, cls, via_ancestor = _innermost_workspace_origin(node, workspace, path_to_module)
        if cls is None:
            continue
        if via_ancestor:
            ancestor_only_classes.add(cls.__name__)
        key = (origin, cls)
        shape, dtype = _node_shape(node)
        src_lines, start, end, src_file, abs_file, varnames = get_cls_meta(cls)
        line_no = _node_line(node, abs_file)
        var = varnames.get(line_no) if line_no is not None else None
        tensor_name = f"{var} [{_node_op_name(node)}]" if var is not None else node.name
        t = TensorInfo(
            name=tensor_name,
            shape=shape,
            annotated_shape=_annotate(shape, val_to_names),
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
                    annotated_shape=_annotate(tuple(p.shape), val_to_names),
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

    # ── Fallback pass: symbolic_trace for workspace classes missed by export ─────
    # Two cases need the fallback:
    #   1. Class not in modules at all — torch.export couldn't trace it.
    #   2. Class in ancestor_only_classes — appeared only via child-module
    #      parent-walk, so its own intermediate ops are missing.
    traced_class_names: set[str] = {m.class_name for m in modules}
    needs_fallback: set[str] = ancestor_only_classes | (
        {
            type(sub_mod).__name__
            for sub_mod in path_to_module.values()
            if _is_workspace_cls(type(sub_mod), workspace)
        } - traced_class_names
    )

    # Pick one representative sub-module instance per class that needs fallback.
    gap: dict[str, tuple[str, nn.Module]] = {}
    for origin, sub_mod in path_to_module.items():
        cls = type(sub_mod)
        if (
            _is_workspace_cls(cls, workspace)
            and cls.__name__ in needs_fallback
            and cls.__name__ not in gap
        ):
            gap[cls.__name__] = (origin, sub_mod)

    if gap:
        # Capture forward() inputs for each gap class by temporarily patching the
        # class's forward method. This intercepts even direct .forward() calls
        # (which bypass __call__ and therefore bypass register_forward_hook).
        raw_captures: dict[str, tuple] = {}
        patched_forwards: dict[type, callable] = {}

        def _make_patched_forward(cls_name: str, target_instance: nn.Module, orig_fwd):
            def _patched(self_inner, *args, **kwargs):
                if cls_name not in raw_captures and self_inner is target_instance:
                    raw_captures[cls_name] = args
                return orig_fwd(self_inner, *args, **kwargs)
            return _patched

        for cls_name, (_origin, sub_mod) in gap.items():
            cls = type(sub_mod)
            if cls not in patched_forwards:
                orig_fwd = cls.forward
                patched_forwards[cls] = orig_fwd
                cls.forward = _make_patched_forward(cls_name, sub_mod, orig_fwd)

        try:
            with torch.no_grad():
                module(*example_args)
        except Exception:
            pass

        for cls, orig_fwd in patched_forwards.items():
            cls.forward = orig_fwd

        for cls_name, (origin, sub_mod) in gap.items():
            input_args = raw_captures.get(cls_name)
            if input_args is None:
                continue
            tensors = _symbolic_trace_fallback(
                sub_mod, input_args, workspace, val_to_names, get_cls_meta
            )
            if not tensors:
                continue
            cls = type(sub_mod)
            src_lines, start, end, src_file, abs_file, varnames = get_cls_meta(cls)
            params = [
                TensorInfo(
                    name=pname,
                    shape=tuple(p.shape),
                    annotated_shape=_annotate(tuple(p.shape), val_to_names),
                    dtype=p.dtype,
                    source_file=src_file,
                    line_number=_param_line(src_lines, start, pname) if start is not None else None,
                )
                for pname, p in sub_mod.named_parameters(recurse=False)
            ]
            # Remove any partial ancestor-walk entry so the full symbolic-trace
            # result is the single authoritative entry for this class.
            modules[:] = [m for m in modules if m.class_name != cls_name]
            modules.append(ModuleInfo(
                class_name=cls_name,
                module_origin=origin or "root",
                source_file=src_file,
                line_start=start,
                line_end=end,
                parameters=params,
                tensors=tensors,
            ))
            traced_class_names.add(cls_name)

    # ── Pass 2: attribute FX nodes to workspace standalone functions ────────────
    # torch.export inlines standalone functions, so stack_trace only records the
    # call site in forward(). We parse that line to find which workspace function
    # is being called, then group nodes by function.
    func_seen_keys: list[tuple] = []
    func_key_to_tensors: dict[tuple, list[TensorInfo]] = {}
    func_key_meta: dict[tuple, tuple] = {}  # (abs_file, func_name) -> (rel_file, line_start, line_end)

    for node in gm.graph.nodes:
        call_abs_file, call_line_no = _node_callsite(node, workspace)
        if call_abs_file is None:
            continue

        src_lines = get_src_lines(call_abs_file)
        called_names = _free_func_names_at_line(src_lines, call_line_no)

        for func_name in called_names:
            if func_name not in func_registry:
                continue
            for abs_func_file, rel_func_file, line_start, line_end in func_registry[func_name]:
                key = (abs_func_file, func_name)
                if key not in func_key_meta:
                    func_key_meta[key] = (rel_func_file, line_start, line_end)

                shape, dtype = _node_shape(node)
                t = TensorInfo(
                    name=node.name,
                    shape=shape,
                    annotated_shape=_annotate(shape, val_to_names),
                    dtype=dtype,
                    source_file=rel_func_file,
                    line_number=None,  # inlined — exact body line not available
                )
                if key not in func_key_to_tensors:
                    func_seen_keys.append(key)
                    func_key_to_tensors[key] = []
                func_key_to_tensors[key].append(t)

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
