"""
llm_annotate.py — LLM-powered rewrite of nn.Module source using shape metadata.

EINSUM mode — converts matrix operations to torch.einsum with mandatory inline comments.
COMMENT mode — deprecated; use EINSUM instead.

Each class is processed in its own async loop (up to max_turns). After each LLM rewrite
the result is validated by running the rewritten forward() against captured inputs and
comparing to the original output. On failure the error is fed back so the LLM can
self-correct. Fixtures and test files are only written for classes that pass.
"""
from __future__ import annotations

import asyncio
import copy
import difflib
import enum
import importlib.util
import os
import subprocess
import tempfile
import traceback
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple, Optional

import torch
import torch.nn as nn
from loguru import logger
from openai import AsyncOpenAI
from tqdm import tqdm

from agent_shaper.fx_utils.get_fx_data import ModuleInfo, get_module_shapes
from agent_shaper.fx_utils.manual_annotate import _annotate_source_lines, _build_line_map
from agent_shaper.fx_utils.test_generator import (
    _CaptureEntry,
    _run_capture_indexed,
    _save_fixtures,
    _TestSpec,
    generate_tests,
)


class _TaskSpec(NamedTuple):
    src_file: str
    line_start: int
    line_end: int
    class_name: str
    class_src: str


class _ValidationResult(NamedTuple):
    passed: bool
    error_msg: str


class _ClassResult(NamedTuple):
    src_file: str
    line_start: int
    line_end: int
    class_name: str
    rewritten_src: str
    passed_validation: bool


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
1. Replace ALL explicit matrix multiplications, batched matmuls, and attention-style
   operations (written with @, torch.matmul, torch.bmm, or equivalent) with
   torch.einsum calls using descriptive index names.
   Do NOT replace nn.Linear / nn.Embedding module calls — leave those as-is.
2. Replace view/transpose/reshape sequences that reshape a tensor to split or merge
   heads (e.g. view(B, T, n_head, d).transpose(1, 2) or the reverse) with the
   equivalent einsum or a single contiguous().view() only where unavoidable.
   Eliminate intermediate reshapes that exist solely to set up a matmul.
3. For every torch.einsum call you produce, add an inline comment that states:
   (a) the full descriptive name for every index letter used, e.g.:
       b=batch_size, t=seq_len, h=n_head, d=head_dim, q=query_pos, k=key_pos,
       v=vocab_size, o=out_features — use full words, not abbreviations, and
   (b) a one-phrase description of what the operation computes, e.g.:
       # b=batch_size, h=n_head, q=query_pos, k=key_pos, d=head_dim; scaled dot-product: (b,h,q,d) x (b,h,k,d) → (b,h,q,k)
   Note: torch.einsum requires single-character subscripts in the equation string —
   the full names belong only in the comment, not in the equation itself.
4. Remove ALL raw-number shape annotations from every line in the class — including
   lines you do not otherwise modify (e.g. gelu, dropout, embedding comments).
   Replace any remaining shape hints with descriptive dimension names only.
5. Everything that cannot be expressed as einsum (layer_norm, softmax, dropout,
   activation functions, nn.Linear calls, embedding lookups) must be preserved
   byte-for-byte. Add or update an inline comment on each such line showing the
   output shape with descriptive dimension names (e.g. # (b, t, n_embd)).

CRITICAL: Your response must be the complete rewritten nn.Module class and nothing else.
No explanation, no markdown fences, no prose before or after the class."""


def _system_prompt(mode: AnnotationMode) -> str:
    return _SYSTEM_COMMENT if mode == AnnotationMode.COMMENT else _SYSTEM_EINSUM


def _outputs_close(a: Any, b: Any, atol: float = 1e-5) -> bool:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.allclose(a.float(), b.float(), atol=atol)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return all(
            _outputs_close(x, y, atol)
            for x, y in zip(a, b)
            if x is not None and y is not None
        )
    return True


def _build_conversation_start(class_src: str, mode: AnnotationMode) -> list[dict]:
    return [
        {"role": "system", "content": _system_prompt(mode)},
        {"role": "user", "content": class_src},
    ]


async def _rewrite_class_turn(
    client: AsyncOpenAI,
    messages: list[dict],
    model: str,
) -> str:
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def _validate_rewrite(
    spec: _TaskSpec,
    rewritten_class_src: str,
    file_source_lines: list[str],
    capture_entry: Optional[_CaptureEntry],
) -> _ValidationResult:
    if capture_entry is None:
        return _ValidationResult(False, "no forward capture available for this class")

    uid = uuid.uuid4().hex[:8]
    temp_path = (
        Path(tempfile.gettempdir()) / f"agent_shaper_{spec.class_name}_{uid}.py"
    )
    try:
        full_src = _apply_replacements(
            file_source_lines,
            [(spec.line_start, spec.line_end, rewritten_class_src)],
        )
        temp_path.write_text(full_src)

        mod_spec = importlib.util.spec_from_file_location(
            f"_tmp_{spec.class_name}_{uid}", temp_path
        )
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
        rewritten_cls = getattr(mod, spec.class_name)

        test_mod = copy.deepcopy(capture_entry.sub_module)
        test_mod.__class__ = rewritten_cls
        test_mod.eval()
        with torch.no_grad():
            actual = test_mod(*capture_entry.input_args)

        if _outputs_close(actual, capture_entry.output):
            return _ValidationResult(True, "")
        return _ValidationResult(False, "outputs differ beyond atol=1e-5")
    except Exception:
        return _ValidationResult(False, traceback.format_exc())
    finally:
        temp_path.unlink(missing_ok=True)


def _append_feedback_messages(
    messages: list[dict],
    original_src: str,
    rewritten_src: str,
    error_msg: str,
) -> None:
    messages.append({"role": "assistant", "content": rewritten_src})
    messages.append({
        "role": "user",
        "content": (
            f"Original module:\n{original_src}\n\n"
            f"Your rewrite:\n{rewritten_src}\n\n"
            f"Validation error:\n{error_msg}\n\n"
            "Fix the forward() so it produces the same outputs as the original."
        ),
    })


async def _rewrite_class_with_validation(
    client: AsyncOpenAI,
    spec: _TaskSpec,
    file_source_lines: list[str],
    capture_entry: Optional[_CaptureEntry],
    tests_dir: Optional[str],
    mode: AnnotationMode,
    model: str,
    max_turns: int,
    pbar: tqdm,
) -> _ClassResult:
    messages = _build_conversation_start(spec.class_src, mode)
    rewritten = spec.class_src
    passed = False
    turns_done = 0

    try:
        for turn in range(1, max_turns + 1):
            rewritten = await _rewrite_class_turn(client, messages, model)
            logger.info(f"{spec.class_name=}, {turn=}")
            result = _validate_rewrite(spec, rewritten, file_source_lines, capture_entry)
            pbar.update(1)
            turns_done += 1

            if result.passed:
                logger.info(f"{spec.class_name=} passed on {turn=}")
                if tests_dir is not None and capture_entry is not None:
                    _save_fixtures(capture_entry, tests_dir, spec.src_file)
                passed = True
                pbar.update(max_turns - turn)
                break

            logger.warning(f"{spec.class_name=} failed on {turn=}, error={result.error_msg!r}")
            _append_feedback_messages(messages, spec.class_src, rewritten, result.error_msg)
    except Exception:
        logger.error(
            f"{spec.class_name=} raised exception:\n{traceback.format_exc()}"
        )
        pbar.update(max_turns - turns_done)

    if not passed:
        logger.warning(f"{spec.class_name=} never passed after {max_turns=}")

    return _ClassResult(
        src_file=spec.src_file,
        line_start=spec.line_start,
        line_end=spec.line_end,
        class_name=spec.class_name,
        rewritten_src=rewritten,
        passed_validation=passed,
    )


def _build_test_specs(
    class_results: list[_ClassResult],
    output_dir: str,
) -> list[_TestSpec]:
    output_dir_rel = os.path.relpath(output_dir)
    return [
        _TestSpec(
            class_name=r.class_name,
            src_file=r.src_file,
            output_dir_rel=output_dir_rel,
            passed_validation=r.passed_validation,
        )
        for r in class_results
    ]


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
    """Return a unified diff between original and rewritten with ANSI colours."""
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
    """Apply (line_start, line_end, new_text) replacements from bottom to top."""
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
    mode: AnnotationMode = AnnotationMode.EINSUM,
    workspace: Optional[str] = None,
    dim_names: Optional[dict[str, int]] = None,
    output_dir: Optional[str] = None,
    show_diff: bool = False,
    open_in_vscode: bool = True,
    tests_dir: Optional[str] = "tests",
    max_turns: int = 3,
) -> dict[str, str]:
    """Rewrite workspace nn.Module classes using an LLM with iterative validation.

    Each class is sent to the LLM, validated against captured forward() outputs, and
    retried up to max_turns if the rewrite is incorrect. Tests and fixtures are only
    written for classes whose rewrite passes validation.

    Returns {relative_source_file: rewritten_source_code}.
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
    specs: list[_TaskSpec] = []
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
        specs.append(
            _TaskSpec(src_file, info.line_start, info.line_end, info.class_name, class_src)
        )

    capture_map = _run_capture_indexed(module, example_args, module_infos)
    tasks: list[asyncio.Task] = []
    pbar = tqdm(total=len(specs) * max_turns, desc="Rewriting modules")

    async with asyncio.TaskGroup() as tg:
        for spec in specs:
            task = tg.create_task(
                _rewrite_class_with_validation(
                    client,
                    spec,
                    file_source_lines.get(spec.src_file, []),
                    capture_map.get(spec.class_name),
                    tests_dir,
                    mode,
                    model,
                    max_turns,
                    pbar,
                )
            )
            tasks.append(task)

    pbar.close()
    class_results: list[_ClassResult] = [t.result() for t in tasks]

    file_replacements: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for r in class_results:
        file_replacements[r.src_file].append((r.line_start, r.line_end, r.rewritten_src))

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

    if output_dir is not None and tests_dir is not None:
        generate_tests(
            _build_test_specs(class_results, output_dir),
            tests_dir=tests_dir,
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
