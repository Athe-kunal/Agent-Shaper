"""
get_fx_data.py
--------------
Export an nn.Module with torch.export, run ShapeProp, and return per-module
tensor shape metadata in topological execution order.
"""

from __future__ import annotations

import inspect
import os
import re
from typing import NamedTuple, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp


class TensorInfo(NamedTuple):
    """Shape metadata for one node in the exported FX graph."""

    name: str
    """FX node name, e.g. 'layer_norm', 'linear_1'."""

    shape: Optional[Union[Tuple[int, ...], Sequence[Optional[Tuple[int, ...]]]]]
    """Output shape — tuple for single tensors, list for multi-output nodes, None otherwise."""

    dtype: Optional[Union[torch.dtype, Sequence[Optional[torch.dtype]]]]
    """Output dtype(s), mirroring shape."""

    line_number: Optional[int]
    """Line in source_filename where this op was called, or None."""


class ModuleCallInfo(NamedTuple):
    """All tensor shapes owned by one nn.Module instance, in topological order."""

    class_name: str
    """Simple class name, e.g. 'CausalSelfAttention'."""

    module_origin: str
    """Qualified instance path, e.g. 'transformer.h.0.attn'.  Empty string = root module."""

    line_start: Optional[int]
    """First line of the class definition in its source file."""

    line_end: Optional[int]
    """Last line of the class definition (inclusive)."""

    source_file: Optional[str]
    """Relative path from cwd to the file that defines the class."""

    tensors: list
    """list[TensorInfo] — every tensor produced by this module instance, in execution order."""


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _parse_tensor_meta(meta):
    if meta is None:
        return None, None
    if hasattr(meta, "shape"):
        return tuple(meta.shape), meta.dtype
    shapes = [tuple(m.shape) if m is not None else None for m in meta]
    dtypes = [m.dtype        if m is not None else None for m in meta]
    return shapes, dtypes


def _source_line(node: torch.fx.Node, filename: str) -> Optional[int]:
    trace = getattr(node, "stack_trace", None) or node.meta.get("stack_trace", "")
    if not trace:
        return None
    hits = re.findall(rf'File "[^"]*{re.escape(filename)}", line (\d+)', trace)
    return int(hits[-1]) if hits else None


def _stack_val_to_instance(val, path_to_module):
    if not (isinstance(val, tuple) and len(val) == 2):
        return None
    first, second = val
    if isinstance(second, nn.Module):
        return second
    if path_to_module is not None and isinstance(first, str):
        return path_to_module.get(first)
    return None


def _module_origin(node: torch.fx.Node, path_to_module) -> Optional[str]:
    stack = node.meta.get("nn_module_stack") or {}
    if not stack:
        return None
    last_val = list(stack.values())[-1]
    if isinstance(last_val, tuple) and len(last_val) == 2:
        qual_path, second = last_val
        if isinstance(qual_path, str) and isinstance(second, str):
            return qual_path
        if isinstance(second, nn.Module):
            return list(stack.keys())[-1]
    return None


def _is_workspace_cls(cls: type, workspace: str) -> bool:
    try:
        f = inspect.getfile(cls)
        return f.startswith(workspace) and "site-packages" not in f
    except (OSError, TypeError):
        return False


def _innermost_workspace_cls(node, root_cls, workspace, path_to_module=None):
    stack = node.meta.get("nn_module_stack") or {}
    for val in reversed(list(stack.values())):
        mod = _stack_val_to_instance(val, path_to_module)
        if mod is not None:
            cls = type(mod)
            if _is_workspace_cls(cls, workspace):
                return cls
    if _is_workspace_cls(root_cls, workspace):
        return root_cls
    return None


# ─── Public API ───────────────────────────────────────────────────────────────

def get_module_shapes(
    module: nn.Module,
    example_args: tuple,
    source_filename: str = "model.py",
    workspace: Optional[str] = None,
) -> list[ModuleCallInfo]:
    """
    Export *module* with torch.export, propagate shapes, and return one
    :class:`ModuleCallInfo` per workspace-defined nn.Module instance encountered
    in the graph — in topological (execution) order.

    Parameters
    ----------
    module:
        nn.Module to analyse (eval mode is set automatically).
    example_args:
        Concrete example inputs matching the module's forward signature.
    source_filename:
        Basename used to extract per-tensor line numbers from stack traces.
    workspace:
        Root directory that defines "your code" vs PyTorch internals.
        Defaults to os.getcwd().

    Returns
    -------
    list[ModuleCallInfo]
        One entry per module instance, ordered by first appearance in the graph.
        Each entry carries the tensor shapes of every node owned by that instance.
    """
    workspace = workspace or os.getcwd()
    root_cls = type(module)
    module = module.eval()
    path_to_module: dict = dict(module.named_modules())

    exported = torch.export.export(module, example_args)
    gm: torch.fx.GraphModule = exported.module()
    ShapeProp(gm).propagate(*example_args)

    seen_keys: list[tuple] = []      # (module_origin, cls) in first-seen order
    key_to_tensors: dict = {}

    for node in gm.graph.nodes:
        cls = _innermost_workspace_cls(node, root_cls, workspace, path_to_module)
        if cls is None:
            continue

        origin = _module_origin(node, path_to_module) or ""
        key = (origin, cls)

        shape, dtype = _parse_tensor_meta(node.meta.get("tensor_meta"))
        tensor = TensorInfo(
            name=node.name,
            shape=shape,
            dtype=dtype,
            line_number=_source_line(node, source_filename),
        )

        if key not in key_to_tensors:
            seen_keys.append(key)
            key_to_tensors[key] = []
        key_to_tensors[key].append(tensor)

    result: list[ModuleCallInfo] = []
    for (origin, cls) in seen_keys:
        try:
            src_lines, start = inspect.getsourcelines(cls)
            src_file = os.path.relpath(inspect.getfile(cls))
        except (OSError, TypeError):
            start, src_file, src_lines = None, None, []
        end = (start + len(src_lines) - 1) if start is not None else None
        result.append(
            ModuleCallInfo(
                class_name=cls.__name__,
                module_origin=origin,
                line_start=start,
                line_end=end,
                source_file=src_file,
                tensors=key_to_tensors[(origin, cls)],
            )
        )

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

    groups = get_module_shapes(GPT(cfg), example_args)

    SEP = "═" * 80
    for g in groups:
        origin_s = g.module_origin or "root"
        print(f"\n{SEP}")
        print(f"  {g.class_name}  [{origin_s}]  L{g.line_start}–L{g.line_end}  {g.source_file}")
        print(SEP)
        for t in g.tensors:
            line_s = f"L{t.line_number}" if t.line_number is not None else "—"
            print(f"  {t.name:<32}  {line_s:^6}  shape={t.shape}  dtype={t.dtype}")
