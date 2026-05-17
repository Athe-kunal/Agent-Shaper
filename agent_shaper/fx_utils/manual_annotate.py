"""
annotate.py — insert tensor shape comments into workspace nn.Module source files.
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

import torch.nn as nn

from agent_shaper.fx_utils.get_fx_data import ModuleInfo, TensorInfo, get_module_shapes


def _shape_str(tensor: TensorInfo) -> str:
    if tensor.annotated_shape is not None:
        return tensor.annotated_shape
    return str(tensor.shape)


def _base_var(name: str) -> str:
    """Return variable name without the trailing ' [op]' suffix."""
    return name.split(" [")[0] if " [" in name else name


def _op_tag(name: str) -> str:
    """Return the op suffix from a name like 'k [transpose]', or '' if none."""
    if " [" in name and name.endswith("]"):
        return name.split(" [", 1)[1][:-1]
    return ""


# Ops that are auxiliary decomposition artifacts from torch.export — not actual
# transformations of the assignment-target tensor (e.g. bias slices inside masked_fill).
_AUXILIARY_OPS = frozenset({
    "slice", "select", "getitem", "index", "index_put",
    "eq", "ne", "lt", "gt", "le", "ge",
    "full", "scalar_tensor", "arange", "zeros", "ones",
    "expand", "expand_as", "clone",
})


def _format_comment(tensors: list[TensorInfo]) -> str:
    """Format a list of TensorInfo at the same line into a comment string.

    For each group sharing the same base variable:
    - Filters out auxiliary decomposition ops (bias slices, comparisons, etc.)
      that torch.export injects but don't represent the variable's transformation.
    - Deduplicates consecutive identical shapes (e.g. contiguous() no-ops).
    - Prepends the input shape of the first op so the full transformation is visible.
    Groups are separated by '  |  '.
    """
    parts: list[str] = []
    i = 0
    while i < len(tensors):
        base = _base_var(tensors[i].name)
        chain = [tensors[i]]
        j = i + 1
        while j < len(tensors) and _base_var(tensors[j].name) == base:
            chain.append(tensors[j])
            j += 1

        # Filter auxiliary ops, but always keep the last node (actual assignment result).
        filtered = [
            t for t in chain[:-1]
            if _op_tag(t.name) not in _AUXILIARY_OPS
        ] + [chain[-1]]

        # Build shape sequence: prepend input shape of first op, then output shapes.
        shapes: list[str] = []
        first = filtered[0]
        if first.input_annotated_shape and first.input_annotated_shape != _shape_str(first):
            shapes.append(first.input_annotated_shape)
        for t in filtered:
            s = _shape_str(t)
            if not shapes or shapes[-1] != s:  # deduplicate consecutive same shapes
                shapes.append(s)

        if len(shapes) > 1:
            parts.append(f"{base}: {' → '.join(shapes)}")
        else:
            shape_str = shapes[0] if shapes else _shape_str(chain[-1])
            # Use plain name (no op tag) when nothing interesting to chain.
            parts.append(f"{base}: {shape_str}")

        i = j
    return "  |  ".join(parts)


def _hashable_shape(shape) -> object:
    if shape is None:
        return None
    if isinstance(shape, list):
        return tuple(tuple(s) if s is not None else None for s in shape)
    return shape


def _build_line_map(
    infos: list,  # list[ModuleInfo | FunctionInfo]
) -> dict[tuple[str, int], list[TensorInfo]]:
    """Return {(rel_source_file, lineno): [TensorInfo]} across all module/function infos."""
    result: dict[tuple[str, int], list[TensorInfo]] = defaultdict(list)
    seen: set[tuple] = set()

    for info in infos:
        for tensor in getattr(info, "parameters", []) + info.tensors:
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


def _annotate_source_lines(
    source_lines: list[str],
    line_annotations: dict[int, list[TensorInfo]],
) -> str:
    """Append shape comments to the relevant lines of a source file."""
    out: list[str] = []
    for i, line in enumerate(source_lines):
        tensors = line_annotations.get(i + 1)
        if tensors:
            stripped = line.rstrip("\n").rstrip()
            comment = _format_comment(tensors)
            out.append(f"{stripped}  # {comment}\n")
        else:
            out.append(line)
    return "".join(out)


def annotate_module_source(
    module: nn.Module,
    example_args: tuple,
    workspace: Optional[str] = None,
    dim_names: Optional[dict[str, int]] = None,
    output_dir: Optional[str] = None,
) -> dict[str, str]:
    """Annotate workspace source files with inline tensor shape comments.

    Returns {relative_source_file: annotated_source_code}.
    If output_dir is given, writes each annotated file under that directory,
    preserving the relative path structure.
    """
    module_infos = get_module_shapes(
        module, example_args, workspace=workspace, dim_names=dim_names
    )
    line_map = _build_line_map(module_infos)

    # Group by file
    file_to_annotations: dict[str, dict[int, list[TensorInfo]]] = defaultdict(dict)
    for (src_file, lineno), tensors in line_map.items():
        file_to_annotations[src_file][lineno] = tensors

    annotated: dict[str, str] = {}
    for src_file, line_annotations in file_to_annotations.items():
        try:
            with open(src_file) as fh:
                source_lines = fh.readlines()
        except OSError:
            continue

        annotated_src = _annotate_source_lines(source_lines, line_annotations)
        annotated[src_file] = annotated_src

        if output_dir is not None:
            out_path = os.path.join(output_dir, src_file)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as fh:
                fh.write(annotated_src)

    return annotated


if __name__ == "__main__":
    import torch
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
    annotated = annotate_module_source(
        GPT(cfg),
        example_args,
        dim_names=dim_names,
        output_dir="annotated_output",
    )

    SEP = "═" * 80
    for path, src in annotated.items():
        print(f"\n{SEP}")
        print(f"  {path}")
        print(SEP)
        print(src)
