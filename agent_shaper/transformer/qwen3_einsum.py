"""
qwen3.py re-expressed with einsum and einops.

Matrix contractions use einops.einsum (esum); structural reshapes/transposes
use einops.rearrange. Attribute names are preserved for load_state_dict compatibility.

GatedDeltaNet and MoEExperts contain data-dependent recurrent/routing loops —
their projections are einsum-converted but the loop bodies are kept intact.
VisionEncoder contains data-dependent control flow and is kept structurally intact.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, Tuple, List
from einops import rearrange
from einops import einsum as esum


# ── Configs ────────────────────────────────────────────────────────────────────

@dataclass
class VisionConfig:
    n_embed: int
    n_layer: int
    n_heads: int
    n_output_embed: int
    n_mlp: int
    num_position_embeddings: int

    in_channels: int = 3
    temporal_patch_size: int = 2
    patch_size: int = 16
    spatial_merge_size: int = 2


@dataclass
class ModelConfig:
    n_embed: int
    n_heads: int
    n_kv_heads: int
    n_layer: int
    n_mlp: int

    n_vocab: int
    tie_word_embeddings: bool

    rope_theta: float
    rms_norm_eps: float
    image_token_id: int = 151655

    d_head: Optional[int] = None
    n_experts: Optional[int] = None
    n_experts_per_token: Optional[int] = None
    n_moe_mlp: Optional[int] = None
    n_shared_expert_mlp: Optional[int] = None

    layer_types: Optional[List[str]] = None
    n_linear_k_heads: Optional[int] = None
    n_linear_v_heads: Optional[int] = None
    d_linear_k: Optional[int] = None
    d_linear_v: Optional[int] = None
    linear_conv_kernel: int = 4
    partial_rotary_factor: float = 1.0


# ── Vision encoder ─────────────────────────────────────────────────────────────

class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return esum(seq, self.inv_freq, "s, d -> s d")


class PatchEmbed(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.n_embed = config.n_embed

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        stride = [self.temporal_patch_size, self.patch_size, self.patch_size]

        self.proj = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.n_embed,
            kernel_size=kernel_size,
            stride=stride,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(x).view(-1, self.n_embed)


class PatchMerger(nn.Module):
    def __init__(self, config: VisionConfig, use_postshuffle_norm: bool = False) -> None:
        super().__init__()
        self.hidden_size = config.n_embed * (config.spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else config.n_embed, eps=1e-6
        )
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.n_output_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(
            x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x
        ).view(-1, self.hidden_size)
        h = self.act_fn(
            esum(x, self.linear_fc1.weight, "n h, m h -> n m") + self.linear_fc1.bias
        )
        return esum(h, self.linear_fc2.weight, "n h, m h -> n m") + self.linear_fc2.bias


class VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.n_embed // config.n_heads
        self.qkv = nn.Linear(config.n_embed, config.n_embed * 3, bias=True)
        self.proj = nn.Linear(config.n_embed, config.n_embed)

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def _apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        orig_dtype = tensor.dtype
        tensor = tensor.float()
        cos = freqs.cos().unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
        sin = freqs.sin().unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
        output = (tensor * cos) + (VisionAttention._rotate_half(tensor) * sin)
        return output.to(orig_dtype)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
        rotary_pos_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        N = x.shape[0]
        qkv_proj = esum(x, self.qkv.weight, "n d, o d -> n o") + self.qkv.bias
        q, k, v = rearrange(
            qkv_proj, "n (three h d) -> three n h d", three=3, h=self.n_heads
        ).unbind(0)

        q = self._apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = self._apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = torch.full(
            [1, N, N], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[
                ...,
                cu_seqlens[i - 1] : cu_seqlens[i],
                cu_seqlens[i - 1] : cu_seqlens[i],
            ] = 0

        q = rearrange(q, "n h d -> h n d")
        k = rearrange(k, "n h d -> h n d")
        v = rearrange(v, "n h d -> h n d")

        attn_weights = esum(q, k, "h q d, h k d -> h q k") / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        attn_output = esum(attn_weights, v, "h q k, h k d -> h q d")
        attn_output = rearrange(attn_output, "h n d -> n (h d)")
        return esum(attn_output, self.proj.weight, "n d, o d -> n o") + self.proj.bias


class VisionMLP(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(config.n_embed, config.n_mlp, bias=True)
        self.linear_fc2 = nn.Linear(config.n_mlp, config.n_embed, bias=True)
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x) -> torch.Tensor:
        h = self.act_fn(
            esum(x, self.linear_fc1.weight, "n d, m d -> n m") + self.linear_fc1.bias
        )
        return esum(h, self.linear_fc2.weight, "n m, d m -> n d") + self.linear_fc2.bias


class VisionBlock(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.n_embed, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.n_embed, eps=1e-6)
        self.attn = VisionAttention(config)
        self.mlp = VisionMLP(config)

    def forward(self, x, cu_seqlens, rotary_pos_emb) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(config=config)

        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.n_embed)
        self.num_grid_per_side = int(config.num_position_embeddings ** 0.5)

        self.blocks = nn.ModuleList([VisionBlock(config) for _ in range(config.n_layer)])
        self.merger = PatchMerger(config=config, use_postshuffle_norm=False)
        head_dim = config.n_embed // config.n_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        self.spatial_merge_size = config.spatial_merge_size

    def fast_pos_embed_interpolate(self, d_image: torch.Tensor) -> torch.Tensor:
        grid_ts, grid_hs, grid_ws = d_image[:, 0], d_image[:, 1], d_image[:, 2]
        device = d_image.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=device
        )
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)

        return torch.cat(patch_pos_embeds_permute, dim=0)

    def rot_pos_emb(self, d_image: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        sms = self.spatial_merge_size

        for t, h, w in d_image:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.view(h // sms, sms, w // sms, sms).transpose(1, 2).flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.view(h // sms, sms, w // sms, sms).transpose(1, 2).flatten()

            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = d_image[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        return rotary_pos_emb_full[pos_ids].flatten(1)

    def forward(self, pixels: torch.Tensor, d_image: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(pixels)
        hidden_states = hidden_states + self.fast_pos_embed_interpolate(d_image)

        rotary_pos_emb = self.rot_pos_emb(d_image)
        cu_seqlens = torch.repeat_interleave(
            d_image[:, 1] * d_image[:, 2], d_image[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

        return self.merger(hidden_states)


# ── Normalisation layers ───────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, n_embed, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embed))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return esum(x.to(input_dtype), self.weight, "b t d, d -> b t d")


class GemmaRMSNorm(nn.Module):
    def __init__(self, n_embed, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n_embed))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        # shape-polymorphic (3-D from layernorm, 4-D from q_norm/k_norm) — broadcast
        return ((1.0 + self.weight.float()) * x).to(input_dtype)


class RMSNormGated(nn.Module):
    def __init__(self, n_embed, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embed))
        self.variance_epsilon = eps

    def forward(self, x, gate):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = esum(x.to(input_dtype), self.weight, "n d, d -> n d")
        return x * F.silu(gate.to(torch.float32)).to(input_dtype)


# ── Rotary embedding ───────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.d_head
        t = config.rope_theta
        r = torch.arange(0, d, 2)
        self.register_buffer("inv_freq", 1.0 / (t ** (r / d)).float(), persistent=False)
        self.mrope_section = [24, 20, 20]

    def forward(self, x, position_ids):
        inv_freq = self.inv_freq.to(dtype=torch.float32, device=x.device)
        # outer product: (3, B, d/2) × (3, B, T) → (3, B, T, d/2)
        inv_freq_3d = inv_freq[None, None, :].expand(3, position_ids.shape[1], -1)
        freqs = esum(inv_freq_3d, position_ids.float(), "m b d, m b t -> m b t d")
        freqs = self._apply_interleaved_mrope(freqs, self.mrope_section)

        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(x.dtype)
        sin = emb.sin().to(x.dtype)
        return cos, sin

    def _apply_interleaved_mrope(self, freqs, mrope_section):
        """[TTT...HHH...WWW] → [THWTHWTHW...TT]"""
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t


# ── Attention ──────────────────────────────────────────────────────────────────

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.n_kv_heads = config.n_kv_heads
        self.n_embed = config.n_embed
        self.partial_rotary_factor = config.partial_rotary_factor

        # q_proj outputs 2x: query + multiplicative gate
        self.q_proj = nn.Linear(self.n_embed, self.n_heads * self.d_head * 2, bias=False)
        self.k_proj = nn.Linear(self.n_embed, self.n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(self.n_embed, self.n_kv_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.d_head, self.n_embed, bias=False)

        self.q_norm = GemmaRMSNorm(self.d_head, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.d_head, eps=config.rms_norm_eps)

    def forward(self, x, cos, sin):
        B, T, _ = x.size()

        qg = rearrange(
            esum(x, self.q_proj.weight, "b t d, o d -> b t o"),
            "b t (h d) -> b t h d", h=self.n_heads,
        )
        q, gate = qg.chunk(2, dim=-1)
        gate = rearrange(gate, "b t h d -> b t (h d)")

        k = rearrange(
            esum(x, self.k_proj.weight, "b t d, o d -> b t o"),
            "b t (h d) -> b t h d", h=self.n_kv_heads,
        )
        k = rearrange(self.k_norm(k), "b t h d -> b h t d")

        v = rearrange(
            esum(x, self.v_proj.weight, "b t d, o d -> b t o"),
            "b t (h d) -> b h t d", h=self.n_kv_heads,
        )

        q = rearrange(self.q_norm(q), "b t h d -> b h t d")
        q, k = self._apply_partial_rotary_pos_emb(q, k, cos, sin)

        if self.n_kv_heads < self.n_heads:
            num_repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(num_repeat, dim=1)
            v = v.repeat_interleave(num_repeat, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = rearrange(y, "b h t d -> b t (h d)")
        y = y * torch.sigmoid(gate)
        return esum(y, self.o_proj.weight, "b t c, d c -> b t d")

    def _apply_partial_rotary_pos_emb(self, q, k, cos, sin):
        rotary_dim = cos.shape[-1]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

        q_rot = (q_rot * cos) + (self._rotate_half(q_rot) * sin)
        k_rot = (k_rot * cos) + (self._rotate_half(k_rot) * sin)

        return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


# ── Linear attention ───────────────────────────────────────────────────────────

class GatedDeltaNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_k_heads = config.n_linear_k_heads
        self.n_v_heads = config.n_linear_v_heads
        self.d_k = config.d_linear_k
        self.d_v = config.d_linear_v
        self.key_dim = self.n_k_heads * self.d_k
        self.value_dim = self.n_v_heads * self.d_v
        conv_kernel = config.linear_conv_kernel

        self.in_proj_qkv = nn.Linear(config.n_embed, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(config.n_embed, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(config.n_embed, self.n_v_heads, bias=False)
        self.in_proj_a = nn.Linear(config.n_embed, self.n_v_heads, bias=False)
        self.out_proj = nn.Linear(self.value_dim, config.n_embed, bias=False)

        conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            conv_dim, conv_dim, conv_kernel,
            groups=conv_dim, padding=conv_kernel - 1, bias=False,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.n_v_heads))
        self.A_log = nn.Parameter(torch.empty(self.n_v_heads).uniform_(0, 16).log())
        self.norm = RMSNormGated(self.d_v, eps=config.rms_norm_eps)

    def forward(self, x):
        B, T, _ = x.shape

        qkv = esum(x, self.in_proj_qkv.weight, "b t d, o d -> b t o")
        qkv = F.silu(self.conv1d(qkv.transpose(1, 2))[:, :, :T]).transpose(1, 2)

        z = esum(x, self.in_proj_z.weight, "b t d, v d -> b t v").view(B, T, self.n_v_heads, self.d_v)
        beta = esum(x, self.in_proj_b.weight, "b t d, h d -> b t h").sigmoid()
        g = -self.A_log.float().exp() * F.softplus(
            esum(x, self.in_proj_a.weight, "b t d, h d -> b t h").float() + self.dt_bias
        )

        q, k, v = torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.view(B, T, self.n_k_heads, self.d_k)
        k = k.view(B, T, self.n_k_heads, self.d_k)
        v = v.view(B, T, self.n_v_heads, self.d_v)

        if self.n_v_heads > self.n_k_heads:
            r = self.n_v_heads // self.n_k_heads
            q = q.repeat_interleave(r, dim=2)
            k = k.repeat_interleave(r, dim=2)

        y = self._gated_delta_rule(q, k, v, g, beta)
        y = self.norm(y.reshape(-1, self.d_v), z.reshape(-1, self.d_v))
        return esum(y.view(B, T, -1), self.out_proj.weight, "b t d, o d -> b t o")

    def _gated_delta_rule(self, q, k, v, g, beta):
        out_dtype = q.dtype
        q, k, v, beta, g = [
            x.transpose(1, 2).contiguous().float() for x in (q, k, v, beta, g)
        ]
        q = self._l2norm(q) / (q.shape[-1] ** 0.5)
        k = self._l2norm(k)

        B, H, T, d_k = k.shape
        d_v = v.shape[-1]
        S = torch.zeros(B, H, d_k, d_v, device=v.device, dtype=v.dtype)
        out = torch.zeros_like(v)

        for t in range(T):
            q_t, k_t, v_t = q[:, :, t], k[:, :, t], v[:, :, t]
            g_t = g[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, :, t].unsqueeze(-1)

            S = S * g_t
            delta = (v_t - (S * k_t.unsqueeze(-1)).sum(-2)) * beta_t
            S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            out[:, :, t] = (S * q_t.unsqueeze(-1)).sum(-2)

        return out.transpose(1, 2).contiguous().to(out_dtype)

    @staticmethod
    def _l2norm(x, eps=1e-6):
        return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


# ── MLP layers ─────────────────────────────────────────────────────────────────

class DenseMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embed, config.n_mlp, bias=False)
        self.up_proj = nn.Linear(config.n_embed, config.n_mlp, bias=False)
        self.down_proj = nn.Linear(config.n_mlp, config.n_embed, bias=False)

    def forward(self, x):
        gate = F.silu(esum(x, self.gate_proj.weight, "b t d, m d -> b t m"))
        up = esum(x, self.up_proj.weight, "b t d, m d -> b t m")
        return esum(gate * up, self.down_proj.weight, "b t m, d m -> b t d")


class SharedExpertMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = F.silu(esum(x, self.gate_proj.weight, "n d, m d -> n m"))
        up = esum(x, self.up_proj.weight, "n d, m d -> n m")
        return esum(gate * up, self.down_proj.weight, "n m, d m -> n d")


class MoEExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts
        self.n_embed = config.n_embed
        self.n_moe_mlp = config.n_moe_mlp

        self.gate_up_proj = nn.Parameter(
            torch.empty(self.n_experts, 2 * self.n_moe_mlp, self.n_embed)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.n_experts, self.n_embed, self.n_moe_mlp)
        )

    def forward(
        self,
        x: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        x_out = torch.zeros_like(x)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.n_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.n_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            x_curr = x[token_idx]
            gate_up = esum(x_curr, self.gate_up_proj[expert_idx], "n d, m d -> n m")
            gate, up = gate_up.chunk(2, dim=-1)
            x_curr = esum(F.silu(gate) * up, self.down_proj[expert_idx], "n d, o d -> n o")
            x_curr = x_curr * top_k_weights[token_idx, top_k_pos, None]
            x_out.index_add_(0, token_idx, x_curr.to(x_out.dtype))

        return x_out


class MoEMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embed = config.n_embed
        self.n_moe_mlp = config.n_moe_mlp
        self.n_experts = config.n_experts
        self.top_k = config.n_experts_per_token
        self.shared_expert_dim = getattr(config, "n_shared_expert_mlp", None)
        self.gate = nn.Linear(self.n_embed, self.n_experts, bias=False)
        self.experts = MoEExperts(config)
        self.shared_expert = None
        self.shared_expert_gate = None
        if self.shared_expert_dim:
            self.shared_expert = SharedExpertMLP(
                hidden_size=self.n_embed,
                intermediate_size=self.shared_expert_dim,
            )
            self.shared_expert_gate = nn.Linear(self.n_embed, 1, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        x_flat = x.reshape(-1, self.n_embed)

        router_logits = esum(x_flat, self.gate.weight, "n d, e d -> n e")
        router_logits = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)
        topk_weights = topk_weights.to(router_logits.dtype)
        expert_out = self.experts(x_flat, topk_indices, topk_weights)

        if self.shared_expert is not None and self.shared_expert_gate is not None:
            shared_expert_out = self.shared_expert(x_flat)
            gate_val = esum(x_flat, self.shared_expert_gate.weight, "n d, o d -> n o")
            expert_out = expert_out + torch.sigmoid(gate_val) * shared_expert_out

        return expert_out.view(B, T, self.n_embed)


# ── Transformer block ──────────────────────────────────────────────────────────

class Block(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        n_embed, eps = config.n_embed, config.rms_norm_eps

        layer_type = "full_attention"
        if config.layer_types is not None:
            layer_type = config.layer_types[layer_idx]

        self.layer_type = layer_type
        self.input_layernorm = GemmaRMSNorm(n_embed=n_embed, eps=eps)
        self.post_attention_layernorm = GemmaRMSNorm(n_embed=n_embed, eps=eps)

        if layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config)
        else:
            self.self_attn = SelfAttention(config)

        self.mlp = MoEMLP(config) if config.n_experts else DenseMLP(config)

    def forward(self, x, cos, sin):
        if self.layer_type == "linear_attention":
            x = x + self.linear_attn(self.input_layernorm(x))
        else:
            x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# ── Full model ─────────────────────────────────────────────────────────────────

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.n_vocab, config.n_embed)
        self.rotary_emb = RotaryEmbedding(config)

        self.layers = nn.ModuleList(
            Block(config, layer_idx=i) for i in range(config.n_layer)
        )
        self.norm = GemmaRMSNorm(config.n_embed, eps=config.rms_norm_eps)

    def forward(self, input_embed, vision_embed=None, vision_mask=None, position_ids=None):
        if vision_embed is not None and vision_mask is not None:
            input_embed[vision_mask] = vision_embed

        cos, sin = self.rotary_emb(input_embed, position_ids)
        for layer in self.layers:
            input_embed = layer(input_embed, cos, sin)

        return self.norm(input_embed)


class Qwen3_5(nn.Module):
    def __init__(self, config: ModelConfig, vision_config: Optional[VisionConfig] = None):
        super().__init__()
        self.config = config
        self.vision_config = vision_config

        self.model = nn.Module()
        self.model.language_model = Model(config)
        self.lm_head = None
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.n_embed, config.n_vocab, bias=False)

        if vision_config is not None:
            self.model.visual = VisionEncoder(vision_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixels: Optional[torch.Tensor] = None,
        d_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_embeds = self.model.language_model.embed_tokens(input_ids)
        position_ids = self._get_position_ids(input_ids=input_ids, d_image=d_image)

        if pixels is not None:
            pixels = pixels.to(input_embeds.dtype)
            vision_embed = self.model.visual(pixels=pixels, d_image=d_image)
            image_pad_token = self.config.image_token_id
            vision_mask = input_ids == image_pad_token
            if vision_mask.sum().item() != vision_embed.shape[0]:
                raise RuntimeError(
                    "Vision token/feature mismatch: "
                    f"mask_tokens={vision_mask.sum().item()} "
                    f"vision_features={vision_embed.shape[0]} "
                    f"image_token_id={image_pad_token}"
                )
            output = self.model.language_model(
                input_embed=input_embeds,
                vision_embed=vision_embed,
                vision_mask=vision_mask,
                position_ids=position_ids,
            )
        else:
            output = self.model.language_model(
                input_embed=input_embeds, position_ids=position_ids
            )

        logits = (
            esum(output, self.model.language_model.embed_tokens.weight, "b t d, v d -> b t v")
            if self.lm_head is None
            else esum(output, self.lm_head.weight, "b t d, v d -> b t v")
        )
        return logits

    def _get_position_ids(
        self, input_ids: torch.Tensor, d_image: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T = input_ids.shape
        image_pad_token = self.config.image_token_id

        if d_image is None:
            position_ids = torch.arange(T, dtype=torch.long, device=input_ids.device)
            return position_ids.unsqueeze(0).expand(3, B, -1)

        position_ids = torch.zeros(3, B, T, dtype=torch.long, device=input_ids.device)
        for batch_idx in range(B):
            seq = input_ids[batch_idx]
            text_idx, image_idx, seq_idx = 0, 0, 0
            while seq_idx < T:
                token_id = seq[seq_idx].item()
                if token_id == image_pad_token:
                    text_idx, image_idx, seq_idx = self._emit_image_block(
                        position_ids=position_ids,
                        batch_idx=batch_idx,
                        seq_idx=seq_idx,
                        text_idx=text_idx,
                        image_idx=image_idx,
                        d_image=d_image,
                    )
                else:
                    position_ids[:, batch_idx, seq_idx] = text_idx
                    text_idx, image_idx, seq_idx = text_idx + 1, image_idx, seq_idx + 1

        return position_ids

    def _emit_image_block(
        self,
        position_ids: torch.Tensor,
        batch_idx: int,
        seq_idx: int,
        text_idx: int,
        image_idx: int,
        d_image: torch.Tensor,
        spatial_merge_size: int = 2,
    ) -> Tuple[int, int, int]:
        t_img, h_img, w_img = d_image[image_idx]
        t_img = int(t_img.item())
        h_img = int((h_img // spatial_merge_size).item())
        w_img = int((w_img // spatial_merge_size).item())

        image_token_count = h_img * w_img
        video_token_count = t_img * image_token_count
        for offset in range(video_token_count):
            target_idx = seq_idx + offset
            remaining = offset % image_token_count
            h_pos = remaining // w_img
            w_pos = remaining % w_img

            position_ids[:, batch_idx, target_idx] = text_idx
            position_ids[1, batch_idx, target_idx] = text_idx + h_pos
            position_ids[2, batch_idx, target_idx] = text_idx + w_pos

        return text_idx + 1, image_idx + 1, seq_idx + video_token_count
