"""MCP server exposing annotated FX shape data for workspace PyTorch modules and functions."""

from __future__ import annotations

import os
import sys
import textwrap
import traceback
from typing import Optional

import torch.nn as nn
from mcp.server.fastmcp import FastMCP

from agent_shaper.fx_utils.get_fx_data import FunctionInfo, ModuleInfo, TensorInfo, get_module_shapes

mcp = FastMCP("agent-shaper-fx")

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
    try:
        ns = _exec_script(script)
    except Exception:
        return f"Script execution failed:\n{traceback.format_exc()}"

    model = _find_module(ns)
    if model is None:
        return (
            "No nn.Module found in script namespace. "
            "Assign your model to a variable named `model`."
        )

    example_args = ns.get("example_args")
    if example_args is None:
        return "Variable `example_args` not found in script namespace."
    if not isinstance(example_args, tuple):
        return "`example_args` must be a tuple."

    dim_names: Optional[dict] = ns.get("dim_names")

    try:
        result = get_module_shapes(
            model,
            example_args,
            workspace=_WORKSPACE,
            dim_names=dim_names,
        )
    except Exception:
        return f"FX shape extraction failed:\n{traceback.format_exc()}"

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


if __name__ == "__main__":
    mcp.run()
