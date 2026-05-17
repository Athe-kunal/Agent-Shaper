import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


def window_partition(x, window_size):
    # (B, H, W, C) -> (B*nh*nw, ws, ws, C)
    # annotated: (2,8,8,96) -> (2,2,ws,2,ws,96) -> (2,2,2,ws,ws,96) -> (8,ws,ws,96)
    return rearrange(
        x, "b (nh ws1) (nw ws2) c -> (b nh nw) ws1 ws2 c",
        ws1=window_size, ws2=window_size,
    )


def window_unpartition(windows, window_size, H, W):
    # (B*nh*nw, ws, ws, C) -> (B, H, W, C)
    # annotated: (8,ws,ws,96) -> (2,2,2,ws,ws,96) -> (2,2,ws,2,ws,96) -> (2,8,8,96)
    nh, nw = H // window_size, W // window_size
    B = windows.shape[0] // (nh * nw)
    return rearrange(
        windows, "(b nh nw) ws1 ws2 c -> b (nh ws1) (nw ws2) c",
        b=B, nh=nh, nw=nw,
    )


class WindowAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head      = config.n_head
        self.head_dim    = config.n_embd // config.n_head
        self.window_size = config.window_size
        self.scale       = self.head_dim ** -0.5

        self.qkv       = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj      = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_drop = nn.Dropout(config.dropout)

        self.rel_pos_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size - 1) * (2 * self.window_size - 1), config.n_head)
        )

        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flat = torch.flatten(coords, 1)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_pos_index = relative_coords.sum(-1)
        self.register_buffer("relative_pos_index", relative_pos_index)

    def forward(self, x, mask=None):
        B, H, W, C = x.shape
        ws = self.window_size
        N  = ws * ws

        # (B,H,W,C) -> (num_win, N, C)  annotated: (8,16,96)
        x_win = rearrange(window_partition(x, ws), "w ws1 ws2 c -> w (ws1 ws2) c")
        num_win = x_win.shape[0]

        # (num_win,N,288) -> three x (num_win, n_head, N, head_dim)  annotated: (8,n_head,16,head_dim)
        q, k, v = rearrange(
            self.qkv(x_win),
            "w n (three h d) -> three w h n d",
            three=3, h=self.n_head,
        ).unbind(0)

        # att: (num_win, n_head, N, N)  annotated: (8,n_head,16,16)
        att = torch.einsum("w h i d, w h j d -> w h i j", q, k) * self.scale

        # rel_bias: (N*N, n_head) -> (n_head, N, N)  annotated: (256,n_head) -> (16,16,n_head) -> (n_head,16,16)
        rel_bias = rearrange(
            self.rel_pos_bias_table[self.relative_pos_index.view(-1)],
            "(i j) h -> h i j", i=N, j=N,
        )
        att = att + rel_bias.unsqueeze(0)

        if mask is not None:
            nW  = mask.shape[0]
            att = rearrange(att, "(b nw) h i j -> b nw h i j", b=B, nw=nW)
            att = att + rearrange(mask, "nw i j -> () nw () i j")
            att = rearrange(att, "b nw h i j -> (b nw) h i j")

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # (num_win, n_head, N, head_dim)  annotated: (8,n_head,16,head_dim)
        x_win = torch.einsum("w h i j, w h j d -> w h i d", att, v)

        # merge heads -> (num_win, N, C) -> (num_win, ws, ws, C)  annotated: (8,16,96) -> (8,ws,ws,96)
        x_win = rearrange(x_win, "w h n d -> w n (h d)")
        x_win = self.proj(x_win)
        x_win = rearrange(x_win, "w (ws1 ws2) c -> w ws1 ws2 c", ws1=ws, ws2=ws)

        # (B, H, W, C)  annotated: (2,8,8,96)
        return window_unpartition(x_win, ws, H, W)
