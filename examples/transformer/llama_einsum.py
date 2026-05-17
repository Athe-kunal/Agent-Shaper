from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)  # out: (batch_size, seq_len, dim)


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    m = torch.arange(seq_len, device=device)
    freqs = torch.outer(m, theta).float()
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))  # x_complex: (batch_size, seq_len, n_heads, head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)  # freqs_complex: (1, seq_len, 1, head_dim/2)
    x_rotated = x_complex * freqs_complex  # x_rotated: (batch_size, seq_len, n_heads, head_dim/2)
    x_out = torch.view_as_real(x_rotated)  # x_out: (batch_size, seq_len, n_heads, head_dim/2, 2)
    x_out = x_out.reshape(*x.shape)  # x_out: (batch_size, seq_len, n_heads, head_dim)
    return x_out.type_as(x).to(device)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    # (b, s, kv_heads, d) → (b, s, kv_heads * n_rep, d)
    return rearrange(
        repeat(x, 'b s h d -> b s h r d', r=n_rep),
        'b s h r d -> b s (h r) d',
    )


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

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_complex: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        batch_size, seq_len, _ = x.shape

        xq = rearrange(self.wq(x), 'b s (h d) -> b s h d', h=self.n_heads_q)  # xq: (batch_size, seq_len, n_heads, head_dim)
        xk = rearrange(self.wk(x), 'b s (h d) -> b s h d', h=self.n_kv_heads)  # xk: (batch_size, seq_len, n_kv_heads, head_dim)
        xv = rearrange(self.wv(x), 'b s (h d) -> b s h d', h=self.n_kv_heads)  # xv: (batch_size, seq_len, n_kv_heads, head_dim)

        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)  # xq: (batch_size, seq_len, n_heads, head_dim)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)  # xk: (batch_size, seq_len, n_kv_heads, head_dim)

        cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        keys = cache_k[:batch_size, : start_pos + seq_len]  # keys: (batch_size, seq_len, n_kv_heads, head_dim)
        values = cache_v[:batch_size, : start_pos + seq_len]  # values: (batch_size, seq_len, n_kv_heads, head_dim)

        keys = repeat_kv(keys, self.n_rep)  # keys: (batch_size, seq_len, n_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # values: (batch_size, seq_len, n_heads, head_dim)

        xq = rearrange(xq, 'b s h d -> b h s d')  # xq: (batch_size, n_heads, seq_len, head_dim)
        keys = rearrange(keys, 'b t h d -> b h t d')  # keys: (batch_size, n_heads, seq_len, head_dim)
        values = rearrange(values, 'b t h d -> b h t d')  # values: (batch_size, n_heads, seq_len, head_dim)

        scores = torch.einsum('b h s d, b h t d -> b h s t', xq, keys) / math.sqrt(self.head_dim)  # scores: (batch_size, n_heads, seq_len, seq_len)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)  # scores: (batch_size, n_heads, seq_len, seq_len)

        output = torch.einsum('b h s t, b h t d -> b h s d', scores, values)  # output: (batch_size, n_heads, seq_len, head_dim)
        output = rearrange(output, 'b h s d -> b s (h d)')  # output: (batch_size, seq_len, dim)
        return self.wo(output)  # out: (batch_size, seq_len, dim)


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
        swish = F.silu(self.w1(x))  # swish: (batch_size, seq_len, hidden_dim)
        x_V = self.w3(x)  # x_V: (batch_size, seq_len, hidden_dim)
        x = swish * x_V  # x: (batch_size, seq_len, hidden_dim)
        x = self.w2(x)  # x: (batch_size, seq_len, dim)
        return x


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

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_complex: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex, cache_k, cache_v)  # h: (batch_size, seq_len, dim)
        out = h + self.feed_forward.forward(self.ffn_norm(h))  # out: (batch_size, seq_len, dim)
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

        n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        head_dim = args.dim // args.n_heads
        self.cache_shape = (args.n_layers, args.max_batch_size, args.max_seq_len, n_kv_heads, head_dim)

    def make_cache(self, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate a fresh (cache_k, cache_v) pair for inference."""
        return (
            torch.zeros(self.cache_shape, device=device),
            torch.zeros(self.cache_shape, device=device),
        )

    def forward(self, tokens: torch.Tensor, start_pos: int, cache_k: torch.Tensor, cache_v: torch.Tensor):
        batch_size, seq_len = tokens.shape

        h = self.tok_embeddings(tokens)  # h: (batch_size, seq_len, dim)

        freqs_complex = self.freqs_complex[start_pos : start_pos + seq_len]  # freqs_complex: (seq_len, head_dim/2)

        for i, layer in enumerate(self.layers):
            h = layer(h, start_pos, freqs_complex, cache_k[i], cache_v[i])  # h: (batch_size, seq_len, dim)
        h = self.norm(h)  # h: (batch_size, seq_len, dim)
        output = self.output(h).float()  # output: (batch_size, seq_len, vocab_size)
        return output
