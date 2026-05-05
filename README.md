# Agent Shaper

Agent Shaper extracts per-module tensor shape metadata from any PyTorch `nn.Module` and uses it to annotate source files — either with descriptive shape comments or by rewriting operations as `torch.einsum`.

## How it works

1. **Shape extraction** (`fx_utils/get_fx_data.py`) — runs `torch.export` + `ShapeProp` on your module to capture the shape of every intermediate tensor in every workspace-defined module's forward pass, not just the inputs.
2. **Manual annotation** (`fx_utils/manual_annotate.py`) — inserts the extracted shapes as inline comments at the end of each relevant source line.
3. **LLM annotation** (`fx_utils/llm_annotate.py`) — feeds each manually-annotated module class to an LLM in parallel. Two modes:
   - **COMMENT** — rewrites comments with descriptive dimension names (e.g. `batch_size`, `seq_len`, `n_embd`) and plain-English explanations of each transformation.
   - **EINSUM** — rewrites the entire module replacing matmuls and attention operations with `torch.einsum`, collapsing intermediate reshapes where possible.
4. **Diff viewer** — review LLM changes before accepting them, either in a Streamlit UI or directly in VS Code's native diff editor.

All modules in a file are processed in parallel via `asyncio`. Everything outside the module classes (imports, dataclasses, config objects) is preserved unchanged in the output file.

## Project structure

```
agent_shaper/
  fx_utils/
    get_fx_data.py       # shape extraction via torch.export + ShapeProp
    manual_annotate.py   # inline shape comment insertion
    llm_annotate.py      # LLM-powered rewrite (COMMENT or EINSUM mode)
    diff_viewer.py       # Streamlit diff UI
  transformer/
    model.py             # example GPT model (nanoGPT)
    run_transformer.py   # example forward pass
```

## Installation

Requires Python 3.12+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

### Manual shape annotation

```python
import torch
from agent_shaper.transformer.model import GPT, GPTConfig
from agent_shaper.fx_utils.manual_annotate import annotate_module_source

cfg = GPTConfig(block_size=32, vocab_size=256, n_layer=2, n_head=2, n_embd=64, dropout=0.0, bias=True)
B, T = 2, 16
example_args = (
    torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.long),
    torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.long),
)

annotated = annotate_module_source(
    GPT(cfg),
    example_args,
    dim_names={"B": B, "T": T},
    output_dir="annotated_output",   # writes annotated files here; omit to just get the dict back
)
```

Or run the built-in smoke test directly:

```bash
python -m agent_shaper.fx_utils.manual_annotate
```

Output files are written to `annotated_output/` preserving the original relative path structure.

### LLM annotation

Set environment variables first:

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o
export OPENAI_BASE_URL=https://your-proxy/v1   # optional; omit for default OpenAI
```

```python
import asyncio, torch
from agent_shaper.transformer.model import GPT, GPTConfig
from agent_shaper.fx_utils.llm_annotate import llm_annotate_module_source, AnnotationMode

cfg = GPTConfig(block_size=32, vocab_size=256, n_layer=2, n_head=2, n_embd=64, dropout=0.0, bias=True)
B, T = 2, 16
example_args = (
    torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.long),
    torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.long),
)

async def main():
    annotated = await llm_annotate_module_source(
        GPT(cfg),
        example_args,
        mode=AnnotationMode.COMMENT,   # or AnnotationMode.EINSUM
        dim_names={"B": B, "T": T},
        output_dir="llm_annotated_output",
    )

asyncio.run(main())
```

Or run the built-in smoke test:

```bash
python -m agent_shaper.fx_utils.llm_annotate
```

### Reviewing diffs

By default, after generating the LLM-annotated file, VS Code opens automatically showing a side-by-side diff of the original vs. the rewritten file (`open_in_vscode=True`). Pass `open_in_vscode=False` to suppress this.

You can also use the Streamlit diff viewer for a browser-based review:

```bash
.venv/bin/streamlit run agent_shaper/fx_utils/diff_viewer.py
```

Enter the path to the original file on the left and the generated file (e.g. `llm_annotated_output/agent_shaper/transformer/model.py`) on the right. The viewer renders a syntax-highlighted unified diff.

### `dim_names`

The optional `dim_names` parameter maps symbolic names to their concrete values in the example run. This lets the shape annotations show `(B, T, n_embd)` instead of `(2, 16, 64)`. When two names share the same value (e.g. `B=2` and `n_head=2`), the annotation shows `B/n_head`.

```python
dim_names = {"B": 2, "T": 16, "n_embd": 64, "n_head": 2}
```

### `get_module_shapes` directly

```python
from agent_shaper.fx_utils import get_module_shapes, TensorInfo

module_infos = get_module_shapes(model, example_args, dim_names=dim_names)
for info in module_infos:
    print(info.class_name, info.source_file, info.line_start, info.line_end)
    for t in info.tensors:
        print(" ", t.name, t.shape, t.annotated_shape)
```

Each `ModuleInfo` contains:
- `class_name`, `module_origin`, `source_file`, `line_start`, `line_end`
- `parameters` — list of `TensorInfo` for `nn.Parameter` entries from `__init__`
- `tensors` — list of `TensorInfo` for every intermediate FX node in the forward pass

Repeated module instances with identical shape sequences (e.g. transformer blocks) are deduplicated to one entry.
