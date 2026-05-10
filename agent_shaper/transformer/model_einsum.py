"""
GPT language model re-expressed with einsum and einops.

Numerically identical to model.py. Every tensor contraction is an explicit
named einsum (via einops.einsum) and every reshape/transpose uses
einops.rearrange.  Attribute names are kept identical to model.py so that
state-dicts are interchangeable via load_state_dict.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from einops import einsum as esum

from agent_shaper.transformer.model import GPTConfig


class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        # x: (B, T, C) -> (B, T, C)
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention using einops.

    Contractions:
      QKV project : b t c, d c -> b t d
      Scores      : b h i d, b h j d -> b h i j
      Context     : b h i j, b h j d -> b h i d
      Out project : b t c, d c -> b t d
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.size()
        H = self.n_head

        # QKV projection — einsum: b t c, d c -> b t d  (d = 3C)
        qkv = esum(x, self.c_attn.weight, "b t c, d c -> b t d")
        if self.c_attn.bias is not None:
            qkv = qkv + self.c_attn.bias
        q, k, v = qkv.split(self.n_embd, dim=-1)  # each (B, T, C)

        # Reshape to multi-head — rearrange: (B, T, C) -> (B, H, T, C/H)
        q = rearrange(q, "b t (h d) -> b h t d", h=H)
        k = rearrange(k, "b t (h d) -> b h t d", h=H)
        v = rearrange(v, "b t (h d) -> b h t d", h=H)

        # Attention scores — einsum: b h i d, b h j d -> b h i j
        scale = 1.0 / math.sqrt(k.size(-1))
        att = esum(q, k, "b h i d, b h j d -> b h i j") * scale
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Context vectors — einsum: b h i j, b h j d -> b h i d
        y = esum(att, v, "b h i j, b h j d -> b h i d")

        # Merge heads — rearrange: (B, H, T, C/H) -> (B, T, C)
        y = rearrange(y, "b h t d -> b t (h d)")

        # Output projection — einsum: b t c, d c -> b t d
        y = esum(y, self.c_proj.weight, "b t c, d c -> b t d")
        if self.c_proj.bias is not None:
            y = y + self.c_proj.bias
        y = self.resid_dropout(y)
        return y


class MLP(nn.Module):
    """
    Feed-forward block using einops.

    Contractions:
      Expand   : b t c, d c -> b t d  (d = 4C)
      Contract : b t c, d c -> b t d  (c = 4C, d = C)
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # x: (B, T, C)
        # Expand — einsum: b t c, d c -> b t d  where d = 4C
        x = esum(x, self.c_fc.weight, "b t c, d c -> b t d")
        if self.c_fc.bias is not None:
            x = x + self.c_fc.bias
        x = self.gelu(x)
        # Contract — einsum: b t c, d c -> b t d  where c = 4C, d = C
        x = esum(x, self.c_proj.weight, "b t c, d c -> b t d")
        if self.c_proj.bias is not None:
            x = x + self.c_proj.bias
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        # (B, T, C) throughout — residual stream
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTEinsum(nn.Module):
    """
    GPT re-implemented with einsum/einops; identical interface to GPT.

    lm_head contraction:
      Training   : b t c, v c -> b t v
      Inference  : b c, v c -> b v  (last position only)
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # (T,)

        # Embeddings: lookup (B,T) -> (B,T,C) and (T,) -> (T,C)
        tok_emb = self.transformer.wte(idx)          # (B, T, C)
        pos_emb = self.transformer.wpe(pos)          # (T, C)
        x = self.transformer.drop(tok_emb + pos_emb) # (B, T, C)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # Training: project all positions — einsum: b t c, v c -> b t v
            logits = esum(x, self.lm_head.weight, "b t c, v c -> b t v")
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # Inference: project last position only — einsum: b c, v c -> b v
            x_last = x[:, -1, :]  # (B, C)
            logits = esum(x_last, self.lm_head.weight, "b c, v c -> b v").unsqueeze(1)  # (B, 1, V)
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
