"""Auto-generated: validates rewrite of GemmaRMSNorm."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / 'GemmaRMSNorm'
_REWRITTEN_SRC = Path('/Users/athekunal/DEV/Agent-Shaper/agent_shaper/transformer/qwen3_einsum.py')
_CLASS_NAME = 'GemmaRMSNorm'


def _load_rewritten_class():
    spec = importlib.util.spec_from_file_location("_rewritten_gemmarmsnorm", _REWRITTEN_SRC)
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


def test_gemmarmsnorm_rewrite_matches_original():
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
        f"Rewrite of {_CLASS_NAME} produces different outputs. "
        "Check the rewritten forward() for correctness."
    )
