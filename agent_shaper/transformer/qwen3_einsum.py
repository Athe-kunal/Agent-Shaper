import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List

from einops import einsum, rearrange


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


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


class PatchEmbed(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.n_embed = config.n_embed
        kernel = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.n_embed, kernel_size=kernel, stride=kernel, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        return self.proj(x).view(-1, self.n_embed)


class PatchMerger(nn.Module):
    def __init__(self, config: VisionConfig, use_postshuffle_norm: bool = False) -> None:
        super().__init__()
        self.hidden_size = config.n_embed * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.n_embed, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.n_output_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


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

    def forward(self, x, cu_seqlens=None, rotary_pos_emb=None) -> torch.Tensor:
        S = x.shape[0]
        q, k, v = self.qkv(x).reshape(S, 3, self.n_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = self._apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = self._apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attn_mask = torch.full([1, S, S], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype)
        for i in range(1, len(cu_seqlens)):
            attn_mask[..., cu_seqlens[i-1]:cu_seqlens[i], cu_seqlens[i-1]:cu_seqlens[i]] = 0

        # [S, n_heads, head_dim] -> [n_heads, S, head_dim]
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

        attn_weights = einsum(q, k, 'h s d, h t d -> h s t') / math.sqrt(self.head_dim)
        attn_weights = nn.functional.softmax(attn_weights + attn_mask, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = einsum(attn_weights, v, 'h s t, h t d -> h s d').transpose(0, 1)  # [S, n_heads, head_dim]
        return self.proj(attn_output.reshape(S, -1))


class VisionMLP(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(config.n_embed, config.n_mlp, bias=True)
        self.linear_fc2 = nn.Linear(config.n_mlp, config.n_embed, bias=True)
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class VisionBlock(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.n_embed, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.n_embed, eps=1e-6)
        self.attn = VisionAttention(config)
        self.mlp = VisionMLP(config)

    def forward(self, x, cu_seqlens, rotary_pos_emb) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
        return x + self.mlp(self.norm2(x))


class VisionEncoder(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(config=config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.n_embed)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        self.blocks = nn.ModuleList([VisionBlock(config) for _ in range(config.n_layer)])
        self.merger = PatchMerger(config=config, use_postshuffle_norm=False)
        self.rotary_pos_emb = VisionRotaryEmbedding((config.n_embed // config.n_heads) // 2)
        self.spatial_merge_size = config.spatial_merge_size

    def fast_pos_embed_interpolate(self, d_image: torch.Tensor) -> torch.Tensor:
        """Interpolate learned position embeddings to match image dimensions."""
        grid_ts, grid_hs, grid_ws = d_image[:, 0], d_image[:, 1], d_image[:, 2]
        device = d_image.device
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]
        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)
            hf, wf = h_idxs.int(), w_idxs.int()
            hc = (hf + 1).clip(max=self.num_grid_per_side - 1)
            wc = (wf + 1).clip(max=self.num_grid_per_side - 1)
            dh, dw = h_idxs - hf, w_idxs - wf
            base_h, base_hc = hf * self.num_grid_per_side, hc * self.num_grid_per_side
            indices = [
                (base_h[None].T + wf[None]).flatten(),
                (base_h[None].T + wc[None]).flatten(),
                (base_hc[None].T + wf[None]).flatten(),
                (base_hc[None].T + wc[None]).flatten(),
            ]
            weights = [
                ((1-dh)[None].T * (1-dw)[None]).flatten(),
                ((1-dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1-dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = (pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]).split(
            [h * w for h, w in zip(grid_hs, grid_ws)]
        )

        out = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = (
                pos_embed.repeat(t, 1)
                .view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            out.append(pos_embed)
        return torch.cat(out, dim=0)

    def rot_pos_emb(self, d_image: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        sms = self.spatial_merge_size
        for t, h, w in d_image:
            hpos = torch.arange(h).unsqueeze(1).expand(-1, w).view(h//sms, sms, w//sms, sms).transpose(1, 2).flatten()
            wpos = torch.arange(w).unsqueeze(0).expand(h, -1).view(h//sms, sms, w//sms, sms).transpose(1, 2).flatten()
            pos_ids.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        rotary_full = self.rotary_pos_emb(d_image[:, 1:].max())
        return rotary_full[pos_ids].flatten(1)

    def forward(self, pixels: torch.Tensor, d_image: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(pixels) + self.fast_pos_embed_interpolate(d_image)
        rotary_pos_emb = self.rot_pos_emb(d_image)
        cu_seqlens = F.pad(
            torch.repeat_interleave(d_image[:, 1] * d_image[:, 2], d_image[:, 0]).cumsum(0, dtype=torch.int32),
            (1, 0),
        )
        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
        return self.merger(hidden_states)


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


class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.d_head
        self.register_buffer("inv_freq", 1.0 / (config.rope_theta ** (torch.arange(0, d, 2) / d)).float(), persistent=False)
        self.mrope_section = [24, 20, 20]

    def forward(self, x, position_ids):
        # inv_freq: [d/2]   position_ids: [3, B, T]
        inv_freq = self.inv_freq.to(dtype=torch.float32, device=x.device)
        freqs = einsum(inv_freq, position_ids.float(), 'd, r b t -> r b t d')  # [3, B, T, d/2]
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)         # [B, T, d/2]
        emb = torch.cat([freqs, freqs], dim=-1)                                 # [B, T, d]
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """[TTT...HHH...WWW] -> [THWTHWTHW...TT]"""
        freqs_t = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):
            idx = slice(offset, mrope_section[dim] * 3, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t


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
        return ((1.0 + self.weight.float()) * x).to(input_dtype)


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.n_kv_heads = config.n_kv_heads
        self.n_embed = config.n_embed
        self.partial_rotary_factor = config.partial_rotary_factor

        # q_proj outputs 2x: query + gate
        self.q_proj = nn.Linear(self.n_embed, self.n_heads * self.d_head * 2, bias=False)
        self.k_proj = nn.Linear(self.n_embed, self.n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(self.n_embed, self.n_kv_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.d_head, self.n_embed, bias=False)
        self.q_norm = GemmaRMSNorm(self.d_head, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.d_head, eps=config.rms_norm_eps)

    def forward(self, x, cos, sin):
        # split q_proj into query + gate
        qg = rearrange(self.q_proj(x), 'b t (h d) -> b t h d', h=self.n_heads)   # [B, T, n_heads, d_head*2]
        q, gate = qg.chunk(2, dim=-1)                                              # [B, T, n_heads, d_head] each
        gate = rearrange(gate, 'b t h d -> b t (h d)')                            # [B, T, n_heads*d_head]

        q = rearrange(self.q_norm(q), 'b t h d -> b h t d')                       # [B, n_heads, T, d_head]
        k = rearrange(
            self.k_norm(rearrange(self.k_proj(x), 'b t (h d) -> b t h d', h=self.n_kv_heads)),
            'b t h d -> b h t d',
        )                                                                           # [B, n_kv_heads, T, d_head]
        v = rearrange(self.v_proj(x), 'b t (h d) -> b h t d', h=self.n_kv_heads)

        q, k = self._apply_partial_rotary_pos_emb(q, k, cos, sin)

        if self.n_kv_heads < self.n_heads:
            r = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(r, dim=1)
            v = v.repeat_interleave(r, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)               # [B, n_heads, T, d_head]
        y = rearrange(y, 'b h t d -> b t (h d)')
        return self.o_proj(y * torch.sigmoid(gate))

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
        conv_dim = self.key_dim * 2 + self.value_dim

        self.in_proj_qkv = nn.Linear(config.n_embed, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(config.n_embed, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(config.n_embed, self.n_v_heads, bias=False)
        self.in_proj_a = nn.Linear(config.n_embed, self.n_v_heads, bias=False)
        self.out_proj = nn.Linear(self.value_dim, config.n_embed, bias=False)
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, conv_kernel, groups=conv_dim, padding=conv_kernel-1, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(self.n_v_heads))
        self.A_log = nn.Parameter(torch.empty(self.n_v_heads).uniform_(0, 16).log())
        self.norm = RMSNormGated(self.d_v, eps=config.rms_norm_eps)

    def forward(self, x):
        B, T, _ = x.shape

        qkv = F.silu(self.conv1d(self.in_proj_qkv(x).transpose(1, 2))[:, :, :T]).transpose(1, 2)
        z = rearrange(self.in_proj_z(x), 'b t (h d) -> b t h d', h=self.n_v_heads)
        beta = self.in_proj_b(x).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(x).float() + self.dt_bias)

        q_raw, k_raw, v_raw = torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = rearrange(q_raw, 'b t (h d) -> b t h d', h=self.n_k_heads)   # [B, T, n_k_heads, d_k]
        k = rearrange(k_raw, 'b t (h d) -> b t h d', h=self.n_k_heads)   # [B, T, n_k_heads, d_k]
        v = rearrange(v_raw, 'b t (h d) -> b t h d', h=self.n_v_heads)   # [B, T, n_v_heads, d_v]

        if self.n_v_heads > self.n_k_heads:
            r = self.n_v_heads // self.n_k_heads
            q = q.repeat_interleave(r, dim=2)
            k = k.repeat_interleave(r, dim=2)

        y = self._gated_delta_rule(q, k, v, g, beta)                      # [B, T, n_v_heads, d_v]
        y = self.norm(rearrange(y, 'b t h d -> (b t h) d'), rearrange(z, 'b t h d -> (b t h) d'))
        return self.out_proj(rearrange(y, '(b t h) d -> b t (h d)', b=B, t=T))

    def _gated_delta_rule(self, q, k, v, g, beta):
        """Recurrent gated delta rule with L2-normalized Q, K."""
        out_dtype = q.dtype
        q, k, v, beta, g = [x.transpose(1, 2).contiguous().float() for x in (q, k, v, beta, g)]
        q = self._l2norm(q) / (q.shape[-1] ** 0.5)
        k = self._l2norm(k)

        B, H, T, d_k = k.shape
        S = torch.zeros(B, H, d_k, v.shape[-1], device=v.device, dtype=v.dtype)
        out = torch.zeros_like(v)

        for t in range(T):
            q_t, k_t, v_t = q[:, :, t], k[:, :, t], v[:, :, t]             # [B, H, d_k/d_v]
            g_t = g[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)               # [B, H, 1, 1]
            beta_t = beta[:, :, t].unsqueeze(-1)                             # [B, H, 1]

            S = S * g_t
            delta = (v_t - einsum(S, k_t, 'b h dk dv, b h dk -> b h dv')) * beta_t
            S = S + einsum(k_t, delta, 'b h dk, b h dv -> b h dk dv')
            out[:, :, t] = einsum(S, q_t, 'b h dk dv, b h dk -> b h dv')

        return out.transpose(1, 2).contiguous().to(out_dtype)

    @staticmethod
    def _l2norm(x, eps=1e-6):
        return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


class RMSNorm(nn.Module):
    def __init__(self, n_embed, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embed))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(input_dtype)


class RMSNormGated(nn.Module):
    def __init__(self, n_embed, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embed))
        self.variance_epsilon = eps

    def forward(self, x, gate):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        x = self.weight * x.to(input_dtype)
        return x * F.silu(gate.to(torch.float32)).to(input_dtype)


class DenseMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embed, config.n_mlp, bias=False)
        self.up_proj = nn.Linear(config.n_embed, config.n_mlp, bias=False)
        self.down_proj = nn.Linear(config.n_mlp, config.n_embed, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))   # [B, T, n_mlp]
        up = self.up_proj(x)               # [B, T, n_mlp]
        return self.down_proj(gate * up)   # [B, T, n_embed]


class SharedExpertMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts
        self.n_embed = config.n_embed
        self.n_moe_mlp = config.n_moe_mlp
        self.gate_up_proj = nn.Parameter(torch.empty(self.n_experts, 2 * self.n_moe_mlp, self.n_embed))
        self.down_proj = nn.Parameter(torch.empty(self.n_experts, self.n_embed, self.n_moe_mlp))

    def forward(self, x, top_k_index, top_k_weights) -> torch.Tensor:
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
            gate, up = F.linear(x_curr, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            x_curr = F.linear(F.silu(gate) * up, self.down_proj[expert_idx])
            x_out.index_add_(0, token_idx, (x_curr * top_k_weights[token_idx, top_k_pos, None]).to(x_out.dtype))

        return x_out


class MoEMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embed = config.n_embed
        self.n_experts = config.n_experts
        self.top_k = config.n_experts_per_token
        self.shared_expert_dim = getattr(config, "n_shared_expert_mlp", None)
        self.gate = nn.Linear(self.n_embed, self.n_experts, bias=False)
        self.experts = MoEExperts(config)
        self.shared_expert = None
        self.shared_expert_gate = None
        if self.shared_expert_dim:
            self.shared_expert = SharedExpertMLP(self.n_embed, self.shared_expert_dim)
            self.shared_expert_gate = nn.Linear(self.n_embed, 1, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        x_flat = x.reshape(-1, self.n_embed)
        router_logits = torch.softmax(F.linear(x_flat, self.gate.weight), dim=-1, dtype=torch.float32)
        topk_weights, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)
        topk_weights = (topk_weights / (topk_weights.sum(-1, keepdim=True) + 1e-9)).to(router_logits.dtype)
        expert_out = self.experts(x_flat, topk_indices, topk_weights)
        if self.shared_expert is not None:
            expert_out = expert_out + torch.sigmoid(self.shared_expert_gate(x_flat)) * self.shared_expert(x_flat)
        return expert_out.view(B, T, self.n_embed)


class Block(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        n_embed, eps = config.n_embed, config.rms_norm_eps
        layer_type = config.layer_types[layer_idx] if config.layer_types is not None else "full_attention"
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
        return x + self.mlp(self.post_attention_layernorm(x))


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.n_vocab, config.n_embed)
        self.rotary_emb = RotaryEmbedding(config)
        self.layers = nn.ModuleList(Block(config, layer_idx=i) for i in range(config.n_layer))
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
        self.lm_head = None if config.tie_word_embeddings else nn.Linear(config.n_embed, config.n_vocab, bias=False)
        if vision_config is not None:
            self.model.visual = VisionEncoder(vision_config)

    def forward(self, input_ids, pixels=None, d_image=None) -> torch.Tensor:
        input_embeds = self.model.language_model.embed_tokens(input_ids)
        position_ids = self._get_position_ids(input_ids=input_ids, d_image=d_image)

        if pixels is not None:
            pixels = pixels.to(input_embeds.dtype)
            vision_embed = self.model.visual(pixels=pixels, d_image=d_image)
            vision_mask = input_ids == self.config.image_token_id
            if vision_mask.sum().item() != vision_embed.shape[0]:
                raise RuntimeError(
                    f"Vision token/feature mismatch: mask_tokens={vision_mask.sum().item()} "
                    f"vision_features={vision_embed.shape[0]}"
                )
            output = self.model.language_model(
                input_embed=input_embeds, vision_embed=vision_embed,
                vision_mask=vision_mask, position_ids=position_ids,
            )
        else:
            output = self.model.language_model(input_embed=input_embeds, position_ids=position_ids)

        lm = self.model.language_model
        return output @ lm.embed_tokens.weight.T if self.lm_head is None else self.lm_head(output)

    def _get_position_ids(self, input_ids, d_image=None) -> torch.Tensor:
        B, T = input_ids.shape
        image_pad_token = self.config.image_token_id
        if d_image is None:
            return torch.arange(T, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(3, B, -1)

        position_ids = torch.zeros(3, B, T, dtype=torch.long, device=input_ids.device)
        for batch_idx in range(B):
            seq = input_ids[batch_idx]
            text_idx, image_idx, seq_idx = 0, 0, 0
            while seq_idx < T:
                if seq[seq_idx].item() == image_pad_token:
                    text_idx, image_idx, seq_idx = self._emit_image_block(
                        position_ids, batch_idx, seq_idx, text_idx, image_idx, d_image,
                    )
                else:
                    position_ids[:, batch_idx, seq_idx] = text_idx
                    text_idx, seq_idx = text_idx + 1, seq_idx + 1
        return position_ids

    def _emit_image_block(
        self, position_ids, batch_idx, seq_idx, text_idx, image_idx, d_image, spatial_merge_size=2,
    ) -> Tuple[int, int, int]:
        t_img, h_img, w_img = d_image[image_idx]
        t_img = int(t_img.item())
        h_img = int((h_img // spatial_merge_size).item())
        w_img = int((w_img // spatial_merge_size).item())
        image_token_count = h_img * w_img
        for offset in range(t_img * image_token_count):
            remaining = offset % image_token_count
            position_ids[:, batch_idx, seq_idx + offset] = text_idx
            position_ids[1, batch_idx, seq_idx + offset] = text_idx + remaining // w_img
            position_ids[2, batch_idx, seq_idx + offset] = text_idx + remaining % w_img
        return text_idx + 1, image_idx + 1, seq_idx + t_img * image_token_count
