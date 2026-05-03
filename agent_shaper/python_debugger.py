import os

import debugpy
import torch

from agent_shaper.transformer.model import LayerNorm


def main():
    if os.environ.get("DEBUGPY_WAIT_FOR_ATTACH"):
        debugpy.listen(("localhost", 5678))
        print(
            "Attach debugger to 127.0.0.1:5678, then execution continues.\n"
            "Tip: use frozen-modules off so breakpoints hit reliably, e.g.\n"
            "  DEBUGPY_WAIT_FOR_ATTACH=1 uv run python -Xfrozen_modules=off -m agent_shaper.python_debugger"
        )
        debugpy.wait_for_client()
        debugpy.breakpoint()

    ndim = 64
    ln = LayerNorm(ndim, bias=True)
    x = torch.randn(2, 8, ndim)
    value = ln(x)
    print(value)


if __name__ == "__main__":
    main()
