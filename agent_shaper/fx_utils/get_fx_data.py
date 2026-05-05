"""
get_fx_data.py — per-module tensor shape metadata for workspace nn.Module classes.

Uses torch.export + ShapeProp to capture every intermediate tensor transformation
inside each module's forward pass, not just the inputs.
"""

from __future__ import annotations

import ast
import inspect
import os
import re
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _is_workspace_cls(cls: type, workspace: str) -> bool:
    try:
        f = inspect.getfile(cls)
        return f.startswith(workspace) and "site-packages" not in f
    except (OSError, TypeError):
        return False


def _innermost_workspace_origin(node, workspace, path_to_module):
    """Return (qualified_path, cls) for the innermost workspace module owning this node."""
    stack = node.meta.get("nn_module_stack") or {}
    for key, val in reversed(list(stack.items())):
        if not (isinstance(val, tuple) and len(val) == 2):
            continue
        qual_path, cls_or_str = val
        # torch.export encodes stack as (str_path, str_classname)
        if isinstance(qual_path, str) and isinstance(cls_or_str, str):
            mod = path_to_module.get(qual_path)
            if mod is not None:
                cls = type(mod)
                if _is_workspace_cls(cls, workspace):
                    return qual_path, cls
        elif isinstance(cls_or_str, type) and _is_workspace_cls(cls_or_str, workspace):
            return qual_path, cls_or_str
    return None, None


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

    offset = class_start - 1  # AST lineno is 1-based within the extracted source
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
    """Invert {name: value} → {value: [names]}, preserving insertion order."""
    result: dict[int, list[str]] = defaultdict(list)
    for name, val in dim_names.items():
        result[val].append(name)
    return dict(result)


def _annotate(shape, val_to_names: dict[int, list[str]]) -> Optional[str]:
    """Return a symbolic shape string, e.g. '(B, T, n_embd)'.
    Ambiguous dims (same value, multiple names) are shown as 'B/n_head'."""
    if shape is None or not val_to_names:
        return None
    if isinstance(shape, list):
        return f"[{', '.join(_annotate(s, val_to_names) or '?' for s in shape)}]"
    parts = ["/".join(val_to_names[d]) if d in val_to_names else str(d) for d in shape]
    return f"({', '.join(parts)})"


# ─── Public API ───────────────────────────────────────────────────────────────

def get_module_shapes(
    module: nn.Module,
    example_args: tuple,
    workspace: Optional[str] = None,
    dim_names: Optional[dict[str, int]] = None,
) -> list[ModuleInfo]:
    """
    Parameters
    ----------
    dim_names:
        Optional ``{name: value}`` mapping for symbolic shape annotation, e.g.
        ``{"B": 2, "T": 16, "C": 64}``.  When a value is shared by multiple
        names (e.g. ``{"B": 2, "n_head": 2}``), the annotation shows ``B/n_head``.
        Raw integer shapes are always preserved in ``TensorInfo.shape``.
    """
    workspace = workspace or os.getcwd()
    module = module.eval()
    path_to_module = dict(module.named_modules())
    val_to_names = _build_val_to_names(dim_names) if dim_names else {}

    exported = torch.export.export(module, example_args)
    gm = exported.module()
    ShapeProp(gm).propagate(*example_args)

    meta_cache: dict[type, tuple] = {}

    def get_meta(cls):
        if cls not in meta_cache:
            src_lines, start, end, src_file, abs_file = _cls_meta(cls)
            varnames = _cls_varnames(cls, start)
            meta_cache[cls] = (src_lines, start, end, src_file, abs_file, varnames)
        return meta_cache[cls]

    # Group FX nodes by (origin_path, cls), preserving first-seen order
    seen_keys: list[tuple] = []
    key_to_tensors: dict[tuple, list[TensorInfo]] = {}

    for node in gm.graph.nodes:
        origin, cls = _innermost_workspace_origin(node, workspace, path_to_module)
        if cls is None:
            continue
        key = (origin, cls)
        shape, dtype = _node_shape(node)
        src_lines, start, end, src_file, abs_file, varnames = get_meta(cls)
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
        if key not in key_to_tensors:
            seen_keys.append(key)
            key_to_tensors[key] = []
        key_to_tensors[key].append(t)

    # Deduplicate: same class + identical shape sequence = same module configuration
    result: list[ModuleInfo] = []
    seen_dedup: set[tuple] = set()

    for (origin, cls) in seen_keys:
        tensors = key_to_tensors[(origin, cls)]
        def _hashable(shape):
            if shape is None:
                return None
            if isinstance(shape, list):
                return tuple(tuple(s) if s is not None else None for s in shape)
            return shape

        dedup_key = (cls.__name__, tuple(_hashable(t.shape) for t in tensors))
        if dedup_key in seen_dedup:
            continue
        seen_dedup.add(dedup_key)

        src_lines, start, end, src_file, abs_file, varnames = get_meta(cls)
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

        result.append(ModuleInfo(
            class_name=cls.__name__,
            module_origin=origin or "root",
            source_file=src_file,
            line_start=start,
            line_end=end,
            parameters=params,
            tensors=tensors,
        ))

    return result


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

    # Supply whatever symbolic names are meaningful for your model.
    # Ambiguous values (e.g. B==n_head==2) are shown as "B/n_head".
    dim_names = {"B": B, "T": T}

    groups = get_module_shapes(GPT(cfg), example_args, dim_names=dim_names)

    SEP = "═" * 80
    for g in groups:
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
