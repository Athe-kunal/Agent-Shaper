"""
llm_annotate.py — LLM-powered rewrite of nn.Module source using shape metadata.

Two modes:
  COMMENT — descriptive comments explaining each transformation with symbolic dim names.
  EINSUM  — convert matrix operations to torch.einsum notation.

All modules are processed in parallel via asyncio.
"""
from __future__ import annotations

import asyncio
import difflib
import enum
import os
import subprocess
from collections import defaultdict
from typing import Optional

import torch.nn as nn
from openai import AsyncOpenAI

from agent_shaper.fx_utils.get_fx_data import ModuleInfo, get_module_shapes
from agent_shaper.fx_utils.manual_annotate import _annotate_source_lines, _build_line_map


class AnnotationMode(enum.Enum):
    COMMENT = "comment"
    EINSUM = "einsum"


_SYSTEM_COMMENT = """\
You are a PyTorch tensor shape annotator.

You will receive a Python nn.Module class that has inline shape comments, for example:
    x = self.c_fc(x)  # x [linear.default]: (2, 16, 256)

The numbers in the shapes are concrete dummy values from one example forward pass —
they are NOT fixed constants. Always replace raw numbers with descriptive dimension
names that reflect what the dimension actually represents (e.g. batch_size, seq_len,
n_embd, n_head, head_dim, vocab_size, etc.).

Your task:
1. Return the EXACT same nn.Module class — every line of Python code must be preserved
   byte-for-byte. Do not add, remove, or reorder any statements.
2. Replace the raw-number shape annotations with descriptive dimension names.
3. Add a brief plain-English comment on each significant operation explaining what
   the transformation does and what the output shape represents.

CRITICAL: Your response must be the complete nn.Module class and nothing else.
No explanation, no markdown fences, no prose before or after the class."""

_SYSTEM_EINSUM = """\
You are a PyTorch-to-einsum converter.

You will receive a Python nn.Module class that has inline shape comments, for example:
    y = q @ k.transpose(-2, -1)  # y: (2, 2, 16, 16)

The numbers in the shapes are concrete dummy values from one example forward pass —
they are NOT fixed constants. Always use descriptive index names that reflect what
each dimension represents (e.g. b=batch, t=seq_len, h=n_head, d=head_dim).

Your task — rewrite the entire nn.Module using einsum notation:
1. Replace ALL matrix multiplications, batched matmuls, linear projections, and
   attention-style operations with torch.einsum calls using descriptive index names.
2. Replace view/transpose/reshape sequences that exist solely to set up a matmul
   with the equivalent einsum directly — eliminate the intermediate reshapes where
   einsum makes them unnecessary.
3. Remove the raw-number shape annotations from the original comments; if you keep
   any comment on a line, it must describe the einsum index meaning, not raw numbers.
4. Everything that cannot be expressed as einsum (layer_norm, softmax, dropout,
   activation functions, embedding lookups) must be preserved byte-for-byte.

CRITICAL: Your response must be the complete rewritten nn.Module class and nothing else.
No explanation, no markdown fences, no prose before or after the class."""


def _system_prompt(mode: AnnotationMode) -> str:
    return _SYSTEM_COMMENT if mode == AnnotationMode.COMMENT else _SYSTEM_EINSUM


async def _rewrite_class(
    client: AsyncOpenAI,
    class_src: str,
    mode: AnnotationMode,
    model: str,
) -> str:
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _system_prompt(mode)},
            {"role": "user", "content": class_src},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def _read_source_lines(path: str) -> list[str]:
    with open(path) as fh:
        return fh.readlines()


def _build_annotated_lines(
    source_lines: list[str],
    line_annotations: dict[int, list],
) -> list[str]:
    """Apply manual shape comments to source_lines and return as a line list."""
    annotated = _annotate_source_lines(source_lines, line_annotations)
    return annotated.splitlines(keepends=True)


def _extract_class_src(
    annotated_lines: list[str],
    line_start: int,
    line_end: int,
) -> str:
    return "".join(annotated_lines[line_start - 1 : line_end])


_ANSI_RED = "\033[31m"
_ANSI_GREEN = "\033[32m"
_ANSI_CYAN = "\033[36m"
_ANSI_RESET = "\033[0m"


def _colored_diff(original: str, rewritten: str, filename: str) -> str:
    """Return a unified diff between original and rewritten with ANSI colours.

    Red  (-) lines: original code that was changed or removed.
    Green(+) lines: LLM-rewritten replacements.
    """
    orig_lines = original.splitlines(keepends=True)
    new_lines = rewritten.splitlines(keepends=True)
    diff = difflib.unified_diff(
        orig_lines, new_lines, fromfile=f"original/{filename}", tofile=f"llm/{filename}"
    )
    parts: list[str] = []
    for line in diff:
        if line.startswith("---") or line.startswith("+++"):
            parts.append(f"{_ANSI_CYAN}{line}{_ANSI_RESET}")
        elif line.startswith("-"):
            parts.append(f"{_ANSI_RED}{line}{_ANSI_RESET}")
        elif line.startswith("+"):
            parts.append(f"{_ANSI_GREEN}{line}{_ANSI_RESET}")
        else:
            parts.append(line)
    return "".join(parts)


def _apply_replacements(
    source_lines: list[str],
    replacements: list[tuple[int, int, str]],
) -> str:
    """Apply (line_start, line_end, new_text) replacements from bottom to top.

    Processing in reverse order keeps earlier line numbers valid as we mutate the list.
    """
    lines = list(source_lines)
    for start, end, new_text in sorted(replacements, key=lambda r: r[0], reverse=True):
        new_lines = new_text.splitlines(keepends=True)
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"
        lines[start - 1 : end] = new_lines
    return "".join(lines)


async def llm_annotate_module_source(
    module: nn.Module,
    example_args: tuple,
    mode: AnnotationMode = AnnotationMode.COMMENT,
    workspace: Optional[str] = None,
    dim_names: Optional[dict[str, int]] = None,
    output_dir: Optional[str] = None,
    show_diff: bool = False,
    open_in_vscode: bool = True,
) -> dict[str, str]:
    """Rewrite workspace nn.Module classes using an LLM.

    Each module class is first annotated with manual shape comments, then sent to
    the LLM for rewriting in the chosen mode. All modules are processed in parallel.
    The rest of each source file (imports, dataclasses, etc.) is preserved unchanged.

    Returns {relative_source_file: rewritten_source_code}.
    If output_dir is given, writes files under that directory preserving path structure.
    If show_diff is True, prints a colour-coded unified diff (red = original, green = LLM)
    instead of printing the full rewritten source.
    """
    model = os.environ["OPENAI_MODEL"]
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    module_infos = get_module_shapes(
        module, example_args, workspace=workspace, dim_names=dim_names
    )
    line_map = _build_line_map(module_infos)

    file_to_annotations: dict[str, dict[int, list]] = defaultdict(dict)
    for (src_file, lineno), tensors in line_map.items():
        file_to_annotations[src_file][lineno] = tensors

    file_source_lines: dict[str, list[str]] = {}
    file_annotated_lines: dict[str, list[str]] = {}
    for src_file in file_to_annotations:
        try:
            src_lines = _read_source_lines(src_file)
        except OSError:
            continue
        file_source_lines[src_file] = src_lines
        file_annotated_lines[src_file] = _build_annotated_lines(
            src_lines, file_to_annotations[src_file]
        )

    seen_keys: set[tuple[str, int]] = set()
    pending: list[tuple[str, int, int, asyncio.Task]] = []

    async with asyncio.TaskGroup() as tg:
        for info in module_infos:
            src_file = info.source_file
            if src_file is None or info.line_start is None or info.line_end is None:
                continue
            if src_file not in file_annotated_lines:
                continue
            cls_key = (src_file, info.line_start)
            if cls_key in seen_keys:
                continue
            seen_keys.add(cls_key)

            class_src = _extract_class_src(
                file_annotated_lines[src_file], info.line_start, info.line_end
            )
            task = tg.create_task(_rewrite_class(client, class_src, mode, model))
            pending.append((src_file, info.line_start, info.line_end, task))

    file_replacements: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for src_file, start, end, task in pending:
        file_replacements[src_file].append((start, end, task.result()))

    result: dict[str, str] = {}
    for src_file, replacements in file_replacements.items():
        src_lines = file_source_lines.get(src_file)
        if src_lines is None:
            continue
        original = "".join(src_lines)
        rewritten = _apply_replacements(src_lines, replacements)
        result[src_file] = rewritten

        if show_diff:
            print(_colored_diff(original, rewritten, os.path.basename(src_file)))

        if output_dir is not None:
            out_path = os.path.join(output_dir, src_file)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as fh:
                fh.write(rewritten)
            if open_in_vscode:
                subprocess.run(
                    ["code", "--diff", os.path.abspath(src_file), os.path.abspath(out_path)]
                )

    return result


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

    async def main():
        await llm_annotate_module_source(
            GPT(cfg),
            example_args,
            mode=AnnotationMode.EINSUM,
            dim_names=dim_names,
            output_dir="llm_annotated_output_einsum_diff",
            show_diff=True,
        )

    asyncio.run(main())
