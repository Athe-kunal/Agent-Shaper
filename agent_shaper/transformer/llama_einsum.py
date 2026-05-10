"""
Llama transformer re-expressed with einsum and einops.

Numerically identical to llama.py. Matrix contractions use einops.einsum
and reshapes/transposes use einops.rearrange.
Attribute names are kept identical to llama.py for load_state_dict compatibility.
"""

from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops import einsum as esum

from agent_shaper.transformer.llama import (
    ModelArgs,
    precompute_theta_pos_frequencies,
    apply_rotary_embeddings,
    repeat_kv,
)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # (B, T, D) normed, then scale by weight (D,) via einsum Hadamard along D
        normed = self._norm(x.float()).type_as(x)
        return esum(normed, self.weight, "b t d, d -> b t d")


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V — einsum: b t d, o d -> b t o
        xq = esum(x, self.wq.weight, "b t d, o d -> b t o")
        xk = esum(x, self.wk.weight, "b t d, o d -> b t o")
        xv = esum(x, self.wv.weight, "b t d, o d -> b t o")

        # Reshape to per-head layout — rearrange: (B, T, H*D) -> (B, T, H, D)
        xq = rearrange(xq, "b t (h d) -> b t h d", h=self.n_heads_q)
        xk = rearrange(xk, "b t (h d) -> b t h d", h=self.n_kv_heads)
        xv = rearrange(xv, "b t (h d) -> b t h d", h=self.n_kv_heads)

        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        keys = self.cache_k[:batch_size, : start_pos + seq_len]    # (B, kT, Hkv, D)
        values = self.cache_v[:batch_size, : start_pos + seq_len]  # (B, kT, Hkv, D)

        keys = repeat_kv(keys, self.n_rep)      # (B, kT, H, D)
        values = repeat_kv(values, self.n_rep)  # (B, kT, H, D)

        # Transpose to (B, H, T, D) for attention — rearrange
        xq = rearrange(xq, "b t h d -> b h t d")
        keys = rearrange(keys, "b t h d -> b h t d")
        values = rearrange(values, "b t h d -> b h t d")

        # Attention scores — einsum: b h q d, b h k d -> b h q k
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = esum(xq, keys, "b h q d, b h k d -> b h q k") * scale
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # Context — einsum: b h q k, b h k d -> b h q d
        output = esum(scores, values, "b h q k, b h k d -> b h q d")

        # Merge heads — rearrange: (B, H, T, D) -> (B, T, H*D)
        output = rearrange(output, "b h t d -> b t (h d)")

        # Output projection — einsum: b t c, d c -> b t d
        return esum(output, self.wo.weight, "b t c, d c -> b t d")


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # SwiGLU: silu(w1(x)) * w3(x), then project down with w2
        # einsum: b t d, h d -> b t h  (expand)
        swish = F.silu(esum(x, self.w1.weight, "b t d, h d -> b t h"))
        x_V = esum(x, self.w3.weight, "b t d, h d -> b t h")
        x = swish * x_V
        # einsum: b t h, d h -> b t d  (contract)
        return esum(x, self.w2.weight, "b t h, d h -> b t d")


class EncoderBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_complex
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(
            self.args.dim // self.args.n_heads,
            self.args.max_seq_len * 2,
            device=self.args.device,
        )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"

        h = self.tok_embeddings(tokens)  # (B, T, D)
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)

        # Output projection — einsum: b t d, v d -> b t v
        return esum(h, self.output.weight, "b t d, v d -> b t v").float()
