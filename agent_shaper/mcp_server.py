"""MCP server exposing annotated FX shape data for workspace PyTorch modules and functions."""

from __future__ import annotations

import ast
import copy
import importlib.util
import io
import os
import sys
import tempfile
import textwrap
import traceback
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch.nn as nn
from mcp.server.fastmcp import FastMCP

from agent_shaper.fx_utils.get_fx_data import (
    FunctionInfo,
    ModuleInfo,
    TensorInfo,
    _build_workspace_func_registry,
    get_module_shapes,
)
from agent_shaper.fx_utils.llm_annotate import _TaskSpec, _validate_rewrite
from agent_shaper.fx_utils.manual_annotate import _annotate_source_lines, _build_line_map
from agent_shaper.fx_utils.test_generator import (
    _capture_function_calls,
    _run_capture_indexed,
    _save_fixtures,
    _save_function_fixtures,
)

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

_WORKSPACE = _REPO_ROOT


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


def _resolve_func_infos(
    func_names: list[str],
    traced_functions: list,
) -> tuple[list, list[str]]:
    """Return (func_info_list, still_missing) for requested func_names.

    First tries the traced graph; falls back to a workspace file-system scan so
    that functions called from non-workspace wrappers are still found.
    """
    traced_map = {f.func_name: f for f in traced_functions}
    registry = _build_workspace_func_registry(_WORKSPACE)

    infos: list[FunctionInfo] = []
    missing: list[str] = []
    for name in func_names:
        if name in traced_map:
            infos.append(traced_map[name])
        elif name in registry:
            abs_file, rel_file, line_start, line_end = registry[name][0]
            infos.append(FunctionInfo(
                func_name=name,
                source_file=rel_file,
                line_start=line_start,
                line_end=line_end,
                tensors=[],
            ))
        else:
            missing.append(name)
    return infos, missing


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


def _build_dep_order(modules: list) -> list[str]:
    """Return unique class names in topological order: leaves first, composites last.

    Uses module origin paths to infer containment: if origin_B starts with origin_A,
    then A contains B and must be rewritten after B.
    """
    class_origins: dict[str, list[str]] = defaultdict(list)
    seen_classes: list[str] = []
    for m in modules:
        origin = "" if m.module_origin == "root" else m.module_origin
        if m.class_name not in class_origins:
            seen_classes.append(m.class_name)
        class_origins[m.class_name].append(origin)

    # deps[A] = classes that A depends on (must be rewritten before A)
    deps: dict[str, set[str]] = {cls: set() for cls in seen_classes}
    for cls_a in seen_classes:
        for oa in class_origins[cls_a]:
            for cls_b in seen_classes:
                if cls_a == cls_b:
                    continue
                for ob in class_origins[cls_b]:
                    if (oa == "" and ob != "") or (oa and ob.startswith(oa + ".")):
                        deps[cls_a].add(cls_b)

    in_degree = {cls: len(deps[cls]) for cls in seen_classes}
    dependents: dict[str, list[str]] = {cls: [] for cls in seen_classes}
    for cls, ds in deps.items():
        for d in ds:
            dependents[d].append(cls)

    queue: deque[str] = deque(cls for cls in seen_classes if in_degree[cls] == 0)
    order: list[str] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for dep in dependents[node]:
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)
    order.extend(cls for cls in seen_classes if cls not in set(order))
    return order


def _workspace_module_classes(abs_src_file: str) -> list[str]:
    """Return names of nn.Module subclasses defined in the file (via AST)."""
    try:
        with open(abs_src_file) as fh:
            tree = ast.parse(fh.read())
    except (OSError, SyntaxError):
        return []
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                try:
                    base_str = ast.unparse(base)
                except AttributeError:
                    base_str = ""
                if "Module" in base_str:
                    names.append(node.name)
                    break
    return names


def _ensure_einops_imports(file_src: str, class_src: str) -> str:
    """Inject missing einops imports into file_src based on names used in class_src."""
    needs_rearrange = "rearrange" in class_src and "from einops import rearrange" not in file_src
    needs_esum = "esum" in class_src and "einsum as esum" not in file_src
    needs_einsum = (
        "einsum" in class_src
        and "einsum as esum" not in file_src
        and "from einops import einsum" not in file_src
    )

    inject: list[str] = []
    if needs_rearrange:
        inject.append("from einops import rearrange\n")
    if needs_esum:
        inject.append("from einops import einsum as esum\n")
    elif needs_einsum:
        inject.append("from einops import einsum\n")

    if not inject:
        return file_src

    lines = file_src.splitlines(keepends=True)
    last_import_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            last_import_idx = i
    return "".join(lines[: last_import_idx + 1] + inject + lines[last_import_idx + 1 :])


def _clone_module(sub_module: nn.Module) -> nn.Module:
    """Clone a module via torch.save/load to avoid deepcopy failures on non-leaf tensors."""
    buf = io.BytesIO()
    torch.save(sub_module, buf)
    buf.seek(0)
    return torch.load(buf, weights_only=False)


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

    # Dependency order header — always derived from the full traced graph, not just
    # the requested files, so composite classes that span files are ordered correctly.
    dep_order = _build_dep_order(shape_result.modules)

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

    dep_lines = "\n".join(f"  {i+1}. {cls}" for i, cls in enumerate(dep_order))
    header = f"Rewrite order (leaves → composites — rewrite top-to-bottom):\n{dep_lines}"

    # Warn about workspace nn.Module classes in the requested files that could not
    # be traced by either torch.export or the symbolic_trace fallback.
    all_traced = {m.class_name for m in shape_result.modules}
    missed: list[str] = []
    for abs_f in requested:
        for cls_name in _workspace_module_classes(abs_f):
            if cls_name not in all_traced and cls_name not in missed:
                missed.append(cls_name)

    if missed:
        missed_lines = "\n".join(f"  - {c}" for c in missed)
        warning = (
            "WARNING: The following classes could not be traced by torch.export "
            "or symbolic_trace — no shape annotations available. Rewrite them "
            "manually using context from their callers above:\n" + missed_lines
        )
        header = warning + "\n\n" + header

    return header + "\n\n" + "\n".join(sections)


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

    # Splice the rewritten class into the original file so its existing imports are
    # preserved, then inject any missing einops imports.
    rewrite_lines = rewritten_class_src.splitlines(keepends=True)
    if rewrite_lines and not rewrite_lines[-1].endswith("\n"):
        rewrite_lines[-1] += "\n"
    patched_lines = (
        file_source_lines[: target.line_start - 1]
        + rewrite_lines
        + file_source_lines[target.line_end :]
    )
    patched_src = _ensure_einops_imports("".join(patched_lines), rewritten_class_src)

    uid = uuid.uuid4().hex[:8]
    temp_path = Path(tempfile.gettempdir()) / f"agent_shaper_cls_{uid}.py"
    try:
        temp_path.write_text(patched_src)
        mod_spec = importlib.util.spec_from_file_location(f"_tmp_cls_{uid}", temp_path)
        tmp_mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(tmp_mod)
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        return f"FAIL: Could not load patched file:\n{traceback.format_exc()}"

    rewritten_cls = getattr(tmp_mod, class_name, None)
    temp_path.unlink(missing_ok=True)
    if rewritten_cls is None:
        return f"FAIL: Class '{class_name}' not found in patched module."

    def _close(a, b, atol: float = 1e-5) -> bool:
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return torch.allclose(a.float(), b.float(), atol=atol)
        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            return all(_close(x, y, atol) for x, y in zip(a, b) if x is not None and y is not None)
        return True

    try:
        test_mod = _clone_module(capture_entry.sub_module)
        test_mod.__class__ = rewritten_cls
        test_mod.eval()
        with torch.no_grad():
            actual = test_mod(*capture_entry.input_args, **capture_entry.input_kwargs)
    except Exception:
        return f"FAIL: Forward pass raised:\n{traceback.format_exc()}"

    if _close(actual, capture_entry.output):
        return f"PASS: '{class_name}' rewrite produces identical outputs (atol=1e-5)."
    return f"FAIL: '{class_name}' outputs differ beyond atol=1e-5."


@mcp.tool()
def validate_file_rewrite(
    script: str,
    rewritten_file_src: str,
) -> str:
    """Validate all nn.Module classes in a rewritten file against original outputs in one call.

    Parses rewritten_file_src, finds every class that matches a traced workspace module,
    validates each against captured forward() outputs, and returns a pass/fail summary.

    The script must assign:
      - `model`       : an nn.Module instance to trace
      - `example_args`: a tuple of example tensors matching the model's forward signature
      - `dim_names`   : (optional) dict mapping symbolic dim names to integer values

    Args:
        script: Python source that sets up model, example_args, and optionally dim_names.
        rewritten_file_src: Complete source of the rewritten file (all classes included).

    Returns:
        Summary table of class names with PASS / FAIL status, and a totals line.
    """
    error, model, example_args, shape_result = _run_fx(script)
    if error:
        return error

    try:
        ast.parse(rewritten_file_src)
    except SyntaxError as exc:
        return f"Syntax error in rewritten file: {exc}"

    capture_map = _run_capture_indexed(model, example_args, shape_result.modules)
    known_classes = {m.class_name for m in shape_result.modules}

    # Load the entire rewritten file as a temp module so its own imports are preserved.
    # This avoids the problem of splicing a class that uses aliased einops names (esum,
    # rearrange) into the original file, which lacks those imports.
    uid = uuid.uuid4().hex[:8]
    temp_path = Path(tempfile.gettempdir()) / f"agent_shaper_file_{uid}.py"
    try:
        temp_path.write_text(rewritten_file_src)
        mod_spec = importlib.util.spec_from_file_location(f"_tmp_file_{uid}", temp_path)
        tmp_mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(tmp_mod)
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        return f"Failed to load rewritten file: {exc}"

    def _close(a, b, atol: float = 1e-5) -> bool:
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return torch.allclose(a.float(), b.float(), atol=atol)
        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            return all(_close(x, y, atol) for x, y in zip(a, b) if x is not None and y is not None)
        return True

    rows: list[str] = []
    for class_name in _build_dep_order(shape_result.modules):
        if class_name not in known_classes or not hasattr(tmp_mod, class_name):
            continue
        entry = capture_map.get(class_name)
        if entry is None:
            rows.append(f"  {class_name:<28} SKIP  (no forward capture)")
            continue
        try:
            rewritten_cls = getattr(tmp_mod, class_name)
            test_mod = _clone_module(entry.sub_module)
            test_mod.__class__ = rewritten_cls
            test_mod.eval()
            with torch.no_grad():
                actual = test_mod(*entry.input_args, **entry.input_kwargs)
            if _close(actual, entry.output):
                rows.append(f"  {class_name:<28} PASS")
            else:
                rows.append(f"  {class_name:<28} FAIL  outputs differ beyond atol=1e-5")
        except Exception:
            first_line = traceback.format_exc().strip().splitlines()[-1]
            rows.append(f"  {class_name:<28} FAIL  {first_line}")

    temp_path.unlink(missing_ok=True)

    if not rows:
        return "No matching workspace nn.Module classes found in rewritten file."

    passed = sum(1 for r in rows if " PASS" in r)
    total = len(rows)
    return f"Results ({passed}/{total} passed):\n" + "\n".join(rows)


@mcp.tool()
def save_fixtures(
    script: str,
    class_names: list[str],
    func_names: list[str] = [],
    tests_dir: str = "tests",
) -> str:
    """Capture and save forward() fixtures for nn.Module classes and standalone functions.

    For each class, hooks into forward() and saves:
      {tests_dir}/{src_stem}/fixtures/{ClassName}/module.pt  — serialised sub-module
      {tests_dir}/{src_stem}/fixtures/{ClassName}/input.pt   — captured forward() inputs
      {tests_dir}/{src_stem}/fixtures/{ClassName}/output.pt  — captured forward() outputs

    For each function, monkey-patches it during the forward pass and saves:
      {tests_dir}/{src_stem}/fixtures/{func_name}/input.pt   — captured call inputs
      {tests_dir}/{src_stem}/fixtures/{func_name}/output.pt  — captured call output
      (no module.pt — functions have no state)

    The script must assign:
      - `model`       : an nn.Module instance to trace
      - `example_args`: a tuple of example tensors matching the model's forward signature

    Args:
        script: Python source that sets up model and example_args.
        class_names: List of nn.Module class names to capture fixtures for.
        func_names: List of standalone function names to capture fixtures for.
        tests_dir: Root test directory (default "tests").

    Returns:
        List of fixture paths written, and any names that could not be captured.
    """
    error, model, example_args, shape_result = _run_fx(script)
    if error:
        return error

    saved: list[str] = []
    missing: list[str] = []

    # --- nn.Module classes ---
    if class_names:
        capture_map = _run_capture_indexed(model, example_args, shape_result.modules)
        class_to_src = {m.class_name: m.source_file for m in shape_result.modules if m.source_file}
        for class_name in class_names:
            entry = capture_map.get(class_name)
            src_file = class_to_src.get(class_name)
            if entry is None or src_file is None:
                missing.append(class_name)
                continue
            _save_fixtures(entry, tests_dir, src_file)
            stem = Path(src_file).stem
            saved.append(f"  {class_name:<28} → {tests_dir}/{stem}/fixtures/{class_name}/")

    # --- standalone functions ---
    if func_names:
        func_infos, unresolved = _resolve_func_infos(func_names, shape_result.functions)
        missing.extend(unresolved)

        fn_capture_map = _capture_function_calls(model, example_args, func_infos)
        for func_name in func_names:
            entry = fn_capture_map.get(func_name)
            if entry is None:
                if func_name not in missing:
                    missing.append(func_name)
                continue
            _save_function_fixtures(entry, tests_dir)
            stem = Path(entry.source_file).stem
            saved.append(f"  {func_name:<28} → {tests_dir}/{stem}/fixtures/{func_name}/")

    lines: list[str] = []
    if saved:
        lines.append("Saved fixtures:")
        lines.extend(saved)
    if missing:
        lines.append("Not found (skipped):")
        lines.extend(f"  {n}" for n in missing)
    return "\n".join(lines) if lines else "No names processed."


def _mcp_test_content(class_name: str, rewritten_abs_path: str) -> str:
    """Generate pytest file content that validates class_name against saved fixtures."""
    lower = class_name.lower()
    rel_src = os.path.relpath(rewritten_abs_path, _WORKSPACE)
    return textwrap.dedent(f"""\
        \"\"\"Auto-generated: validates rewrite of {class_name}.\"\"\"
        from __future__ import annotations

        import importlib.util
        from pathlib import Path

        import torch

        _FIXTURE_DIR = Path(__file__).parent / "fixtures" / {repr(class_name)}
        _REWRITTEN_SRC = Path(__file__).parents[2] / {repr(rel_src)}
        _CLASS_NAME = {repr(class_name)}


        def _load_rewritten_class():
            spec = importlib.util.spec_from_file_location("_rewritten_{lower}", _REWRITTEN_SRC)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, _CLASS_NAME)


        def _outputs_close(a, b, atol: float = 1e-5) -> bool:
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                return torch.allclose(a.float(), b.float(), atol=atol)
            if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
                return all(
                    _outputs_close(x, y, atol)
                    for x, y in zip(a, b)
                    if x is not None and y is not None
                )
            return True


        def test_{lower}_rewrite_matches_original():
            original_module = torch.load(_FIXTURE_DIR / "module.pt", weights_only=False)
            input_args = torch.load(_FIXTURE_DIR / "input.pt", weights_only=False)
            input_kwargs = torch.load(_FIXTURE_DIR / "input_kwargs.pt", weights_only=False)
            expected_output = torch.load(_FIXTURE_DIR / "output.pt", weights_only=False)

            rewritten_cls = _load_rewritten_class()
            original_module.__class__ = rewritten_cls

            original_module.eval()
            with torch.no_grad():
                actual_output = original_module(*input_args, **input_kwargs)

            assert _outputs_close(actual_output, expected_output), (
                f"Rewrite of {{_CLASS_NAME}} produces different outputs. "
                "Check the rewritten forward() for correctness."
            )
    """)


def _mcp_function_test_content(func_name: str, rewritten_abs_path: str) -> str:
    """Generate pytest file content that validates a rewritten standalone function."""
    lower = func_name.lower()
    rel_src = os.path.relpath(rewritten_abs_path, _WORKSPACE)
    return textwrap.dedent(f"""\
        \"\"\"Auto-generated: validates rewrite of {func_name}.\"\"\"
        from __future__ import annotations

        import importlib.util
        import sys
        from pathlib import Path

        import torch

        _FIXTURE_DIR = Path(__file__).parent / "fixtures" / {repr(func_name)}
        _REWRITTEN_SRC = Path(__file__).parents[2] / {repr(rel_src)}
        _FUNC_NAME = {repr(func_name)}


        def _load_rewritten_fn():
            spec = importlib.util.spec_from_file_location("_rewritten_{lower}", _REWRITTEN_SRC)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            return getattr(mod, _FUNC_NAME)


        def _outputs_close(a, b, atol: float = 1e-5) -> bool:
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                return torch.allclose(a.float(), b.float(), atol=atol)
            if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
                return all(
                    _outputs_close(x, y, atol)
                    for x, y in zip(a, b)
                    if x is not None and y is not None
                )
            return True


        def test_{lower}_rewrite_matches_original():
            input_args = torch.load(_FIXTURE_DIR / "input.pt", weights_only=False)
            input_kwargs = torch.load(_FIXTURE_DIR / "input_kwargs.pt", weights_only=False)
            expected_output = torch.load(_FIXTURE_DIR / "output.pt", weights_only=False)

            rewritten_fn = _load_rewritten_fn()
            actual_output = rewritten_fn(*input_args, **input_kwargs)

            assert _outputs_close(actual_output, expected_output), (
                f"Rewrite of {{_FUNC_NAME}} produces different outputs. "
                "Check the rewritten function for correctness."
            )
    """)


@mcp.tool()
def generate_test_files(
    script: str,
    class_names: list[str],
    rewritten_src_file: str,
    func_names: list[str] = [],
    tests_dir: str = "tests",
) -> str:
    """Generate pytest test files that validate rewritten classes and functions against saved fixtures.

    For each class_name, writes:
      {tests_dir}/{orig_src_stem}/test_rewrite_{classname}.py
      (loads the rewritten class, swaps it into the fixture module, asserts output identity)

    For each func_name, writes:
      {tests_dir}/{orig_src_stem}/test_rewrite_{func_name}.py
      (imports the rewritten function directly, calls it with fixture inputs, asserts output identity)

    Run save_fixtures first to create the fixture files the tests depend on.

    The script must assign:
      - `model`       : an nn.Module instance to trace
      - `example_args`: a tuple of example tensors matching the model's forward signature

    Args:
        script: Python source that sets up model and example_args.
        class_names: List of nn.Module class names to generate tests for.
        rewritten_src_file: Relative or absolute path to the rewritten source file.
        func_names: List of standalone function names to generate tests for.
        tests_dir: Root test directory (default "tests").

    Returns:
        List of test file paths written, and any names not found in the traced graph.
    """
    error, model, example_args, shape_result = _run_fx(script)
    if error:
        return error

    rewritten_abs = os.path.abspath(rewritten_src_file)
    if not os.path.exists(rewritten_abs):
        return f"Rewritten source file not found: {rewritten_abs}"

    written: list[str] = []
    missing: list[str] = []

    # --- nn.Module classes ---
    class_to_src = {m.class_name: m.source_file for m in shape_result.modules if m.source_file}
    for class_name in class_names:
        src_file = class_to_src.get(class_name)
        if src_file is None:
            missing.append(class_name)
            continue
        stem = Path(src_file).stem
        test_dir = Path(tests_dir) / stem
        test_dir.mkdir(parents=True, exist_ok=True)
        content = _mcp_test_content(class_name, rewritten_abs)
        test_path = test_dir / f"test_rewrite_{class_name.lower()}.py"
        test_path.write_text(content)
        written.append(f"  {class_name:<28} → {test_path}")

    # --- standalone functions ---
    resolved_funcs, unresolved_funcs = _resolve_func_infos(func_names, shape_result.functions)
    missing.extend(unresolved_funcs)
    for fi in resolved_funcs:
        func_name = fi.func_name
        src_file = fi.source_file
        if src_file is None:
            missing.append(func_name)
            continue
        stem = Path(src_file).stem
        test_dir = Path(tests_dir) / stem
        test_dir.mkdir(parents=True, exist_ok=True)
        content = _mcp_function_test_content(func_name, rewritten_abs)
        test_path = test_dir / f"test_rewrite_{func_name.lower()}.py"
        test_path.write_text(content)
        written.append(f"  {func_name:<28} → {test_path}")

    lines: list[str] = []
    if written:
        lines.append("Generated test files:")
        lines.extend(written)
        lines.append(
            f"\nExpected fixture layout:  {tests_dir}/{{src_stem}}/fixtures/{{name}}/[input|output].pt"
        )
        lines.append("Run save_fixtures first if fixtures do not exist yet.")
    if missing:
        lines.append("Not found in traced graph (skipped):")
        lines.extend(f"  {n}" for n in missing)
    return "\n".join(lines) if lines else "No names processed."


if __name__ == "__main__":
    mcp.run(transport="stdio")
