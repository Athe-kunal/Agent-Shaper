"""
test_generator.py — auto-generates pytest tests that validate LLM-rewritten nn.Module classes.

For each workspace nn.Module that passes iterative validation, generates:
  tests/{parent_module}/fixtures/{ClassName}/module.pt   — serialized original sub-module
  tests/{parent_module}/fixtures/{ClassName}/input.pt    — captured forward() input tuple
  tests/{parent_module}/fixtures/{ClassName}/output.pt   — captured original forward() output
  tests/{parent_module}/test_rewrite_{classname}.py      — pytest file with relative paths
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, NamedTuple

import torch
import torch.nn as nn

from agent_shaper.fx_utils.get_fx_data import ModuleInfo


class _CaptureEntry(NamedTuple):
    class_name: str
    origin: str
    sub_module: nn.Module
    input_args: tuple
    input_kwargs: dict
    output: Any


class _TestSpec(NamedTuple):
    class_name: str
    src_file: str        # relative, e.g. "agent_shaper/transformer/model.py"
    output_dir_rel: str  # relative, e.g. "llm_annotated_output_einsum"
    passed_validation: bool


class _ForwardHook:
    """Captures the first (input_args, output) seen for a given class name."""

    def __init__(self, class_name: str, captures: dict) -> None:
        self._class_name = class_name
        self._captures = captures

    def __call__(self, _module: nn.Module, args: tuple, kwargs: dict, output: Any) -> None:
        if self._class_name not in self._captures:
            self._captures[self._class_name] = (args, kwargs, output)


def _detach(obj: Any) -> Any:
    """Recursively detach and CPU-move tensors; pass through everything else."""
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, (tuple, list)):
        return type(obj)(_detach(x) for x in obj)
    return obj


def _run_capture(
    module: nn.Module,
    example_args: tuple,
    module_infos: list[ModuleInfo],
) -> list[_CaptureEntry]:
    """Run a forward pass with hooks to record one (input, output) per unique class."""
    path_to_module = dict(module.named_modules())
    raw_captures: dict[str, tuple] = {}
    origin_by_class: dict[str, str] = {}
    handles: list = []

    for info in module_infos:
        if info.class_name in origin_by_class:
            continue
        raw_origin = "" if info.module_origin == "root" else info.module_origin
        sub_mod = path_to_module.get(raw_origin)
        if sub_mod is None:
            continue
        origin_by_class[info.class_name] = raw_origin
        hook = _ForwardHook(info.class_name, raw_captures)
        handles.append(sub_mod.register_forward_hook(hook, with_kwargs=True))

    with torch.no_grad():
        module.eval()
        module(*example_args)

    for handle in handles:
        handle.remove()

    entries = []
    for class_name, (input_args, input_kwargs, output) in raw_captures.items():
        raw_origin = origin_by_class[class_name]
        sub_mod = path_to_module.get(raw_origin, module)
        entries.append(_CaptureEntry(
            class_name=class_name,
            origin=raw_origin,
            sub_module=sub_mod,
            input_args=_detach(input_args),
            input_kwargs={k: _detach(v) for k, v in input_kwargs.items()},
            output=_detach(output),
        ))

    return entries


def _run_capture_indexed(
    module: nn.Module,
    example_args: tuple,
    module_infos: list[ModuleInfo],
) -> dict[str, _CaptureEntry]:
    """Return {class_name: _CaptureEntry} for all captured workspace classes."""
    return {e.class_name: e for e in _run_capture(module, example_args, module_infos)}


def _parent_module_name(src_file: str) -> str:
    return Path(src_file).stem


def _save_fixtures(entry: _CaptureEntry, tests_dir: str, src_file: str) -> None:
    fdir = (
        Path(tests_dir)
        / _parent_module_name(src_file)
        / "fixtures"
        / entry.class_name
    )
    fdir.mkdir(parents=True, exist_ok=True)
    torch.save(entry.sub_module, fdir / "module.pt")
    torch.save(entry.input_args, fdir / "input.pt")
    torch.save(entry.input_kwargs, fdir / "input_kwargs.pt")
    torch.save(entry.output, fdir / "output.pt")


def _generate_test_content(
    class_name: str,
    src_file_rel: str,
    output_dir_rel: str,
) -> str:
    lower = class_name.lower()
    return textwrap.dedent(f"""\
        \"\"\"Auto-generated: validates LLM rewrite of {class_name}.\"\"\"
        from __future__ import annotations

        import importlib.util
        from pathlib import Path

        import torch

        _FIXTURE_DIR = Path(__file__).parent / "fixtures" / {repr(class_name)}
        _REWRITTEN_SRC = (
            Path(__file__).parent.parent.parent
            / {repr(output_dir_rel)}
            / {repr(src_file_rel)}
        )
        _CLASS_NAME = {repr(class_name)}


        def _load_rewritten_class():
            spec = importlib.util.spec_from_file_location(
                "_rewritten_{lower}", _REWRITTEN_SRC
            )
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
            original_module = torch.load(
                _FIXTURE_DIR / "module.pt", weights_only=False
            )
            input_args = torch.load(_FIXTURE_DIR / "input.pt", weights_only=False)
            input_kwargs = torch.load(_FIXTURE_DIR / "input_kwargs.pt", weights_only=False)
            expected_output = torch.load(
                _FIXTURE_DIR / "output.pt", weights_only=False
            )

            rewritten_cls = _load_rewritten_class()
            original_module.__class__ = rewritten_cls

            original_module.eval()
            with torch.no_grad():
                actual_output = original_module(*input_args, **input_kwargs)

            assert _outputs_close(actual_output, expected_output), (
                f"LLM rewrite of {{_CLASS_NAME}} produces different outputs. "
                "Check the rewritten forward() for correctness."
            )
    """)


def generate_tests(test_specs: list[_TestSpec], tests_dir: str = "tests") -> None:
    """Write pytest files for each class that passed validation."""
    for spec in test_specs:
        if not spec.passed_validation:
            continue
        parent = _parent_module_name(spec.src_file)
        test_dir = Path(tests_dir) / parent
        test_dir.mkdir(parents=True, exist_ok=True)
        content = _generate_test_content(
            spec.class_name, spec.src_file, spec.output_dir_rel
        )
        test_file = test_dir / f"test_rewrite_{spec.class_name.lower()}.py"
        test_file.write_text(content)
