import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_unpartition(windows, window_size, H, W):
    B_times_num_wins = windows.shape[0]
    B = B_times_num_wins // ((H // window_size) * (W // window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head     = config.n_head
        self.head_dim   = config.n_embd // config.n_head
        self.window_size = config.window_size
        self.scale      = self.head_dim ** -0.5

        self.qkv  = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_drop = nn.Dropout(config.dropout)

        # relative position bias table
        self.rel_pos_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size - 1) * (2 * self.window_size - 1), config.n_head)
        )

        # precompute relative position index
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords   = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
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

        x_win   = window_partition(x, ws)
        N       = ws * ws
        x_win   = x_win.view(-1, N, C)
        num_win = x_win.shape[0]

        qkv = self.qkv(x_win)
        qkv = qkv.reshape(num_win, N, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        att = (q @ k.transpose(-2, -1)) * self.scale

        # relative position bias
        rel_bias = self.rel_pos_bias_table[self.relative_pos_index.view(-1)]
        rel_bias = rel_bias.view(N, N, self.n_head)
        rel_bias = rel_bias.permute(2, 0, 1).contiguous()
        att = att + rel_bias.unsqueeze(0)

        if mask is not None:
            nW  = mask.shape[0]
            att = att.view(B, nW, self.n_head, N, N)
            att = att + mask.unsqueeze(1).unsqueeze(0)
            att = att.view(-1, self.n_head, N, N)

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        x_win = (att @ v)
        x_win = x_win.transpose(1, 2)
        x_win = x_win.reshape(num_win, N, C)
        x_win = self.proj(x_win)
        x_win = x_win.view(-1, ws, ws, C)

        x_out = window_unpartition(x_win, ws, H, W)
        return x_out