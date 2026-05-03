"""
get_fx_data.py
--------------
Export a callable nn.Module with torch.export, run ShapeProp, and return
per-node information as a list of FxNodeInfo NamedTuples for downstream use.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, NamedTuple, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp


# ─── Return type ─────────────────────────────────────────────────────────────

class FxNodeInfo(NamedTuple):
    """All extracted metadata for a single node in the exported FX graph."""

    name: str
    """FX node name (e.g. 'layer_norm', 'split', 'linear_1')."""

    op: str
    """FX op kind: placeholder | get_attr | call_function | call_method | output."""

    target: Any
    """The concrete callable / attribute string that this node invokes."""

    source_line: Optional[int]
    """Line number in the originating source file, or None if not available."""

    output_shape: Optional[Union[Tuple[int, ...], Sequence[Optional[Tuple[int, ...]]]]]
    """
    Output tensor shape(s).
    - Single tensor  → tuple[int, ...]
    - Multi-output   → list[tuple[int, ...] | None]   (None for non-tensor slots)
    - Non-tensor op  → None
    """

    output_dtype: Optional[Union[torch.dtype, Sequence[Optional[torch.dtype]]]]
    """Mirrors output_shape but carries the dtype(s)."""

    module_origin: Optional[str]
    """
    Innermost nn.Module class name + extra_repr() that owns this op.
    e.g. 'Linear(in_features=64, out_features=192, bias=True)'
    """

    constants: Sequence[Any]
    """
    Non-node (Python literal) arguments baked into this op.
    e.g. for layer_norm → [[64], 1e-05];  for split → [64, 2].
    """

    module_line_start: Optional[int]
    """
    First line of the innermost nn.Module *class* in its source file.
    e.g. CausalSelfAttention starts at line 29 in model.py.
    """

    module_line_end: Optional[int]
    """Last line of the innermost nn.Module class (inclusive)."""

    module_source_file: Optional[str]
    """Absolute path to the file that defines the innermost nn.Module class."""

    module_source: Optional[str]
    """Full source code of the innermost nn.Module class."""


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _parse_tensor_meta(
    meta: Any,
) -> Tuple[
    Optional[Union[Tuple[int, ...], list]],
    Optional[Union[torch.dtype, list]],
]:
    """Decompose a node's tensor_meta into (shape, dtype)."""
    if meta is None:
        return None, None
    if hasattr(meta, "shape"):                          # single TensorMetadata
        return tuple(meta.shape), meta.dtype
    # immutable_list — multi-output node (e.g. aten.split, tuple returns)
    shapes = [tuple(m.shape) if m is not None else None for m in meta]
    dtypes = [m.dtype        if m is not None else None for m in meta]
    return shapes, dtypes


def _source_line(node: torch.fx.Node, filename: str = "model.py") -> Optional[int]:
    """Return the innermost line number in *filename* from node.stack_trace."""
    trace = getattr(node, "stack_trace", None) or node.meta.get("stack_trace", "")
    if not trace:
        return None
    hits = re.findall(rf'File "[^"]*{re.escape(filename)}", line (\d+)', trace)
    return int(hits[-1]) if hits else None


def _module_origin(node: torch.fx.Node) -> Optional[str]:
    """
    Innermost nn.Module from nn_module_stack formatted as
    'ClassName(extra_repr)'.
    """
    stack = node.meta.get("nn_module_stack") or {}
    if not stack:
        return None
    cls_name, mod_instance = list(stack.values())[-1]
    try:
        r = mod_instance.extra_repr()
        return f"{cls_name}({r})" if r else cls_name
    except Exception:
        return cls_name


def _literal_args(node: torch.fx.Node) -> list:
    """Collect non-node (Python literal) values from node.args and node.kwargs."""
    parts: list = []
    for a in node.args:
        if not isinstance(a, torch.fx.Node):
            parts.append(a)
    for k, v in (node.kwargs or {}).items():
        if not isinstance(v, torch.fx.Node):
            parts.append(v)          # callers can use (k, v) pairs if needed
    return parts


def _module_source_info(
    node: torch.fx.Node,
    root_module: Optional[nn.Module] = None,
) -> Tuple[Optional[int], Optional[int], Optional[str], Optional[str]]:
    """
    Return (line_start, line_end, source_file, source_code) for the innermost
    nn.Module class that owns this node, using inspect on the class definition.

    ``nn_module_stack`` is empty when the node belongs to the root module itself
    (torch.export only tracks submodule ancestry).  In that case *root_module*
    is used as the fallback so root-level exports still get source info.

    Nodes such as get_attr / placeholder that carry no module attribution
    return (None, None, None, None).
    """
    stack = node.meta.get("nn_module_stack") or {}
    if stack:
        _, mod_instance = list(stack.values())[-1]
        cls = type(mod_instance)
    elif root_module is not None:
        cls = type(root_module)
    else:
        return None, None, None, None

    try:
        source_lines, start_lineno = inspect.getsourcelines(cls)
        source_file = inspect.getfile(cls)
    except (OSError, TypeError):
        return None, None, None, None

    end_lineno = start_lineno + len(source_lines) - 1
    source_code = "".join(source_lines)
    return start_lineno, end_lineno, source_file, source_code


# ─── Public API ───────────────────────────────────────────────────────────────

def get_fx_data(
    module: nn.Module,
    example_args: tuple,
    source_filename: str = "model.py",
) -> list[FxNodeInfo]:
    """
    Export *module* with torch.export, propagate shapes, and return per-node
    metadata as a list of :class:`FxNodeInfo` NamedTuples.

    Parameters
    ----------
    module:
        An ``nn.Module`` in eval mode.  If not already in eval mode the
        function will call ``.eval()`` on a copy — the caller's instance is
        left unchanged.
    example_args:
        A tuple of concrete example tensors that match the module's ``forward``
        signature (same role as in ``torch.export.export``).
    source_filename:
        Basename used when scanning ``node.stack_trace`` for source line
        numbers.  Defaults to ``"model.py"``.

    Returns
    -------
    list[FxNodeInfo]
        One entry per node in the exported FX graph, in sequential execution
        order (topological order as produced by torch.export).
    """
    module = module.eval()

    exported = torch.export.export(module, example_args)
    gm: torch.fx.GraphModule = exported.module()

    ShapeProp(gm).propagate(*example_args)

    results: list[FxNodeInfo] = []
    for node in gm.graph.nodes:            # already in sequential call order
        shape, dtype                          = _parse_tensor_meta(node.meta.get("tensor_meta"))
        line_start, line_end, src_file, source = _module_source_info(node, root_module=module)

        results.append(
            FxNodeInfo(
                name                = node.name,
                op                  = node.op,
                target              = node.target,
                source_line         = _source_line(node, source_filename),
                output_shape        = shape,
                output_dtype        = dtype,
                module_origin       = _module_origin(node),
                constants           = _literal_args(node),
                module_line_start   = line_start,
                module_line_end     = line_end,
                module_source_file  = src_file,
                module_source       = source,
            )
        )

    return results


# ─── CLI smoke-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from agent_shaper.transformer.model import (
        GPT, GPTConfig,
        LayerNorm, CausalSelfAttention, MLP, Block,
    )

    cfg = GPTConfig(
        block_size=32,
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
        bias=True,
    )

    B, T, C = 2, 16, cfg.n_embd

    modules_to_test = [
        ("LayerNorm",           LayerNorm(C, bias=True),         (torch.rand(B, T, C),)),
        ("CausalSelfAttention", CausalSelfAttention(cfg),        (torch.rand(B, T, C),)),
        ("MLP",                 MLP(cfg),                        (torch.rand(B, T, C),)),
        ("Block",               Block(cfg),                      (torch.rand(B, T, C),)),
        ("GPT",                 GPT(cfg),                        (
            torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.long),
            torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.long),  # targets
        )),
    ]

    SEP = "═" * 100

    for display_name, module, args in modules_to_test:
        nodes = get_fx_data(module, args)

        # Module-level info — same for every node, grab from first non-None hit
        mod_info = next((n for n in nodes if n.module_line_start is not None), None)
        if mod_info is not None:
            rel_path = os.path.relpath(mod_info.module_source_file)
            line_range = f"L{mod_info.module_line_start}–L{mod_info.module_line_end}"
        else:
            rel_path, line_range = "—", "—"

        print(f"\n{SEP}")
        print(f"  {display_name}  —  {len(nodes)} nodes  ({line_range})  {rel_path}")
        print(SEP)

        # Per-node table: tensor-relevant columns only
        print(f"  {'node':<38}  {'call':^6}  {'shape':<32}  {'dtype':<18}  constants")
        print(f"  {'-'*38}  {'-'*6}  {'-'*32}  {'-'*18}  ---------")
        for n in nodes:
            line_s   = f"L{n.source_line}" if n.source_line  is not None else "—"
            shape_s  = str(n.output_shape) if n.output_shape is not None else "—"
            dtype_s  = str(n.output_dtype) if n.output_dtype is not None else "—"
            consts_s = repr(n.constants)   if n.constants                 else "—"
            print(f"  {n.name:<38}  {line_s:^6}  {shape_s:<32}  {dtype_s:<18}  {consts_s}")
