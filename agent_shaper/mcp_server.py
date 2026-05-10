"""MCP server exposing annotated FX shape data for workspace PyTorch modules and functions."""

from __future__ import annotations

import os
import sys
import textwrap
import traceback
from collections import defaultdict
from typing import Optional

import torch.nn as nn
from mcp.server.fastmcp import FastMCP

from agent_shaper.fx_utils.get_fx_data import FunctionInfo, ModuleInfo, TensorInfo, get_module_shapes
from agent_shaper.fx_utils.llm_annotate import _TaskSpec, _validate_rewrite
from agent_shaper.fx_utils.manual_annotate import _annotate_source_lines, _build_line_map
from agent_shaper.fx_utils.test_generator import _run_capture_indexed

mcp = FastMCP("agent-shaper-fx")


@mcp.prompt()
def rewrite_workflow() -> str:
    """Describes the required workflow for rewriting nn.Module classes."""
    return """\
Rewrite workflow — follow these steps in order for every nn.Module or function you rewrite:

1. Call `get_annotated_sources` with the setup script and the list of relevant source files.
   This returns the original source with inline shape comments showing concrete tensor shapes.

2. Rewrite the class using torch.einsum for contractions and einops for structural axis
   operations, following the shape comments. Produce the complete rewritten class source.

3. MANDATORY: call `validate_rewrite` with the same setup script, the class name, and
   your rewritten source BEFORE reporting the rewrite as complete.
   - If it returns "PASS", the rewrite is correct — report success.
   - If it returns "FAIL", read the error message, fix the rewritten source, and call
     `validate_rewrite` again. Repeat until it passes or you exhaust your attempts.

Never skip step 3. A rewrite that has not passed `validate_rewrite` must not be
presented to the user as a finished result."""

_WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _format_tensor(t: TensorInfo, indent: str = "    ") -> str:
    ann = f"  {t.annotated_shape}" if t.annotated_shape else ""
    loc = f"  {t.source_file}:L{t.line_number}" if t.source_file and t.line_number else ""
    return f"{indent}{t.name:<32}  shape={t.shape}{ann}{loc}"


def _format_module(m: ModuleInfo) -> str:
    sep = "─" * 72
    lines = [
        sep,
        f"  [module] {m.class_name}  [{m.module_origin}]  L{m.line_start}–{m.line_end}  {m.source_file}",
        sep,
    ]
    if m.parameters:
        lines.append("  Parameters:")
        lines.extend(_format_tensor(t) for t in m.parameters)
    if m.tensors:
        lines.append("  Tensors:")
        lines.extend(_format_tensor(t) for t in m.tensors)
    return "\n".join(lines)


def _format_function(f: FunctionInfo) -> str:
    sep = "─" * 72
    lines = [
        sep,
        f"  [function] {f.func_name}  L{f.line_start}–{f.line_end}  {f.source_file}",
        sep,
    ]
    if f.tensors:
        lines.append("  Tensors:")
        lines.extend(_format_tensor(t) for t in f.tensors)
    return "\n".join(lines)


def _exec_script(script: str) -> dict:
    """Execute script and return its local namespace."""
    ns: dict = {"__builtins__": __builtins__}
    if _WORKSPACE not in sys.path:
        sys.path.insert(0, _WORKSPACE)
    exec(textwrap.dedent(script), ns)  # noqa: S102
    return ns


def _find_module(ns: dict) -> Optional[nn.Module]:
    """Return the first nn.Module found in namespace, preferring 'model'."""
    if "model" in ns and isinstance(ns["model"], nn.Module):
        return ns["model"]
    for val in ns.values():
        if isinstance(val, nn.Module):
            return val
    return None


def _run_fx(script: str):
    """Execute the script and run FX tracing. Returns (error_str, model, example_args, shape_result)."""
    try:
        ns = _exec_script(script)
    except Exception:
        return f"Script execution failed:\n{traceback.format_exc()}", None, None, None

    model = _find_module(ns)
    if model is None:
        return (
            "No nn.Module found in script namespace. "
            "Assign your model to a variable named `model`.",
            None, None, None,
        )

    example_args = ns.get("example_args")
    if example_args is None:
        return "Variable `example_args` not found in script namespace.", None, None, None
    if not isinstance(example_args, tuple):
        return "`example_args` must be a tuple.", None, None, None

    dim_names: Optional[dict] = ns.get("dim_names")

    try:
        shape_result = get_module_shapes(
            model, example_args, workspace=_WORKSPACE, dim_names=dim_names
        )
    except Exception:
        return f"FX shape extraction failed:\n{traceback.format_exc()}", None, None, None

    return None, model, example_args, shape_result


@mcp.tool()
def get_fx_shapes(
    script: str,
    module_filter: Optional[str] = None,
    function_filter: Optional[str] = None,
) -> str:
    """Run a PyTorch setup script and return annotated FX shape data for workspace modules and functions.

    The script must assign:
      - `model`       : an nn.Module instance to trace
      - `example_args`: a tuple of example tensors matching the model's forward signature
      - `dim_names`   : (optional) dict mapping symbolic dim names to integer values

    Args:
        script: Python source that sets up model, example_args, and optionally dim_names.
        module_filter: Optional class name (e.g. "CausalSelfAttention") to restrict module output.
        function_filter: Optional function name (e.g. "scaled_dot_product") to restrict function output.

    Returns:
        Formatted string listing workspace modules and standalone functions with annotated tensor shapes.
    """
    error, model, example_args, result = _run_fx(script)
    if error:
        return error

    modules = result.modules
    functions = result.functions

    if module_filter:
        modules = [m for m in modules if m.class_name == module_filter]
    if function_filter:
        functions = [f for f in functions if f.func_name == function_filter]

    sections = []
    if modules:
        sections.append("\n".join(_format_module(m) for m in modules))
    elif module_filter:
        sections.append(f"No module named '{module_filter}' found after tracing.")

    if functions:
        sections.append("\n".join(_format_function(f) for f in functions))
    elif function_filter:
        sections.append(f"No function named '{function_filter}' found after tracing.")

    if not sections:
        return "No workspace modules or functions found in the traced graph."

    return "\n".join(sections)


@mcp.tool()
def get_annotated_sources(
    script: str,
    files: list[str],
) -> str:
    """Run a PyTorch setup script and return inline-shape-annotated source for specified files.

    The script must assign:
      - `model`       : an nn.Module instance to trace
      - `example_args`: a tuple of example tensors matching the model's forward signature
      - `dim_names`   : (optional) dict mapping symbolic dim names to integer values

    FX tracing is run on the model, then each workspace nn.Module class and standalone
    function whose source file matches one of the requested `files` is extracted and
    returned with inline shape comments appended to every assignment line.

    Args:
        script: Python source that sets up model, example_args, and optionally dim_names.
        files: Relative paths of source files to scan (e.g. ["agent_shaper/transformer/model.py"]).

    Returns:
        Annotated source snippets for every matching module and function, with shape
        comments showing tensor shapes at each assignment line.

    IMPORTANT: After rewriting any class returned here, you MUST call `validate_rewrite`
    to confirm the rewrite produces identical outputs before reporting it as complete.
    See the `rewrite_workflow` prompt for the full required sequence.
    """
    error, model, example_args, shape_result = _run_fx(script)
    if error:
        return error

    requested = {os.path.abspath(f) for f in files}

    # Build per-file line→tensors map from the full shape result.
    line_map = _build_line_map(shape_result.modules)
    file_to_annotations: dict[str, dict[int, list]] = defaultdict(dict)
    for (src_file, lineno), tensors in line_map.items():
        if os.path.abspath(src_file) in requested:
            file_to_annotations[src_file][lineno] = tensors

    modules = [
        m for m in shape_result.modules
        if m.source_file and os.path.abspath(m.source_file) in requested
    ]
    functions = [
        f for f in shape_result.functions
        if f.source_file and os.path.abspath(f.source_file) in requested
    ]

    if not modules and not functions:
        return "No workspace modules or functions found in the specified files."

    # Read and annotate each relevant source file once.
    file_annotated_lines: dict[str, list[str]] = {}
    all_src_files = (
        {m.source_file for m in modules} | {f.source_file for f in functions}
    ) - {None}
    for src_file in all_src_files:
        try:
            with open(src_file) as fh:
                source_lines = fh.readlines()
        except OSError:
            continue
        annotated = _annotate_source_lines(
            source_lines, file_to_annotations.get(src_file, {})
        )
        file_annotated_lines[src_file] = annotated.splitlines(keepends=True)

    sep = "─" * 72
    sections: list[str] = []
    seen: set[tuple[str, int]] = set()

    for info in modules:
        if info.source_file not in file_annotated_lines:
            continue
        key = (info.source_file, info.line_start)
        if key in seen:
            continue
        seen.add(key)
        ann_lines = file_annotated_lines[info.source_file]
        snippet = "".join(ann_lines[info.line_start - 1 : info.line_end])
        sections.append(
            f"{sep}\n"
            f"  [module] {info.class_name}  "
            f"{info.source_file}:L{info.line_start}–{info.line_end}\n"
            f"{sep}\n"
            f"{snippet}"
        )

    for info in functions:
        if info.source_file not in file_annotated_lines:
            continue
        key = (info.source_file, info.line_start)
        if key in seen:
            continue
        seen.add(key)
        ann_lines = file_annotated_lines[info.source_file]
        snippet = "".join(ann_lines[info.line_start - 1 : info.line_end])
        sections.append(
            f"{sep}\n"
            f"  [function] {info.func_name}  "
            f"{info.source_file}:L{info.line_start}–{info.line_end}\n"
            f"{sep}\n"
            f"{snippet}"
        )

    return "\n".join(sections)


@mcp.tool()
def validate_rewrite(
    script: str,
    class_name: str,
    rewritten_class_src: str,
) -> str:
    """Validate an LLM-rewritten nn.Module class against the original forward() outputs.

    The script must assign:
      - `model`       : an nn.Module instance to trace
      - `example_args`: a tuple of example tensors matching the model's forward signature
      - `dim_names`   : (optional) dict mapping symbolic dim names to integer values

    The original forward() is captured via hooks, then the rewritten class is swapped in
    and run against the same inputs. Outputs are compared with atol=1e-5.

    Args:
        script: Python source that sets up model, example_args, and optionally dim_names.
        class_name: Name of the nn.Module class that was rewritten (e.g. "CausalSelfAttention").
        rewritten_class_src: Complete source of the rewritten class (einsum/einops version).

    Returns:
        "PASS: ..." on success, or "FAIL: ..." with a detailed error message.
    """
    error, model, example_args, shape_result = _run_fx(script)
    if error:
        return error

    target = next(
        (m for m in shape_result.modules if m.class_name == class_name), None
    )
    if target is None:
        return f"Class '{class_name}' not found in traced workspace modules."
    if target.source_file is None or target.line_start is None or target.line_end is None:
        return f"Source location not available for class '{class_name}'."

    capture_map = _run_capture_indexed(model, example_args, shape_result.modules)
    capture_entry = capture_map.get(class_name)
    if capture_entry is None:
        return f"Could not capture forward() output for class '{class_name}'."

    try:
        with open(target.source_file) as fh:
            file_source_lines = fh.readlines()
    except OSError:
        return f"Could not read source file: {target.source_file}"

    spec = _TaskSpec(
        src_file=target.source_file,
        line_start=target.line_start,
        line_end=target.line_end,
        class_name=class_name,
        class_src=rewritten_class_src,
    )
    result = _validate_rewrite(spec, rewritten_class_src, file_source_lines, capture_entry)

    if result.passed:
        return f"PASS: '{class_name}' rewrite produces identical outputs (atol=1e-5)."
    return f"FAIL: {result.error_msg}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
