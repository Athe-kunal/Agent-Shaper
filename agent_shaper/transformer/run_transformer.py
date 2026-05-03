import torch

from agent_shaper.transformer.model import GPT, GPTConfig

batch_size = 64
seq_len = 256

cfg = GPTConfig()
model = GPT(cfg)
idx = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), dtype=torch.long)
logits, loss = model(idx)

print(logits.shape, loss)