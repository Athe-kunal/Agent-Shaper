"""Einops-annotated rewrite of opd.py.

Shape key used in comments:
  B  — batch size
  T  — sequence length
  C  — chunk length (C == T when chunk_size <= 0)
  V  — vocabulary size
  K  — top-K budget
"""

from __future__ import annotations

import torch
from einops import rearrange
from typing import Literal


def _chunk_range(T: int, chunk: int):
    """Yield (start, end) pairs that partition [0, T) into slices of size `chunk`."""
    for t0 in range(0, T, chunk):
        yield t0, min(t0 + chunk, T)


def _effective_chunk(T: int, *chunk_sizes: int) -> int:
    """Return the smallest positive chunk_size, or T (no-op) if none are positive."""
    active = [c for c in chunk_sizes if c > 0]
    return min(active) if active else T


def student_topk_indices(
    student_logits: torch.Tensor,   # [B, T, V]
    K: int,
    chunk_size: int = -1,
) -> torch.Tensor:                  # [B, T, K]
    """Select top-K vocab indices from student logits (no grad)."""
    T = student_logits.shape[1]
    chunk = _effective_chunk(T, chunk_size)
    parts = []
    with torch.no_grad():
        for t0, t1 in _chunk_range(T, chunk):
            sl = student_logits[:, t0:t1]           # [B, C, V]
            _, idx = sl.topk(K, dim=-1)              # [B, C, K]
            parts.append(idx)
    return torch.cat(parts, dim=1)                   # [B, T, K]


def teacher_logprobs_at_indices(
    teacher_logits: torch.Tensor,   # [B, T, V]
    topk_idx: torch.Tensor,         # [B, T, K]
    chunk_size: int = -1,
) -> torch.Tensor:                  # [B, T, K]
    """Compute teacher log-probs at pre-selected top-K indices (no grad)."""
    T = teacher_logits.shape[1]
    chunk = _effective_chunk(T, chunk_size)
    parts = []
    with torch.no_grad():
        for t0, t1 in _chunk_range(T, chunk):
            sl = teacher_logits[:, t0:t1]                            # [B, C, V]
            lse = torch.logsumexp(sl, dim=-1)                        # [B, C]
            lp = sl.gather(-1, topk_idx[:, t0:t1]) - rearrange(lse, 'b c -> b c 1')  # [B, C, K]
            parts.append(lp)
    del teacher_logits
    return torch.cat(parts, dim=1)                                   # [B, T, K]


def teacher_topk_logprobs(
    teacher_logits: torch.Tensor,   # [B, T, V]
    K: int,
    chunk_size: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:   # ([B, T, K], [B, T, K])
    """Teacher selects top-K indices and computes its own log-probs (no grad)."""
    T = teacher_logits.shape[1]
    chunk = _effective_chunk(T, chunk_size)
    idx_parts, lp_parts = [], []
    with torch.no_grad():
        for t0, t1 in _chunk_range(T, chunk):
            sl = teacher_logits[:, t0:t1]                            # [B, C, V]
            lse = torch.logsumexp(sl, dim=-1)                        # [B, C]
            _, idx = sl.topk(K, dim=-1)                              # [B, C, K]
            lp = sl.gather(-1, idx) - rearrange(lse, 'b c -> b c 1')  # [B, C, K]
            idx_parts.append(idx)
            lp_parts.append(lp)
    del teacher_logits
    return torch.cat(idx_parts, dim=1), torch.cat(lp_parts, dim=1)  # [B, T, K], [B, T, K]


def student_logprobs_at_indices(
    student_logits: torch.Tensor,   # [B, T, V]
    topk_idx: torch.Tensor,         # [B, T, K]
    chunk_size: int = -1,
) -> torch.Tensor:                  # [B, T, K]
    """Compute student log-probs at given indices, preserving the gradient graph."""
    T = student_logits.shape[1]
    chunk = _effective_chunk(T, chunk_size)
    parts = []
    for t0, t1 in _chunk_range(T, chunk):
        sl = student_logits[:, t0:t1]                                # [B, C, V]
        lse = torch.logsumexp(sl, dim=-1)                            # [B, C]
        lp = sl.gather(-1, topk_idx[:, t0:t1]) - rearrange(lse, 'b c -> b c 1')  # [B, C, K]
        parts.append(lp)
    return torch.cat(parts, dim=1)                                   # [B, T, K]


def _topk_logprobs_slice(
    s: torch.Tensor,   # [B, C, V]  student logits chunk (grad-enabled)
    t: torch.Tensor,   # [B, C, V]  teacher logits chunk (no grad)
    top_k: int,
    select_topk_by: Literal["student", "teacher"],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Core top-K log-prob computation for one [B, C, V] slice."""
    s_lse = torch.logsumexp(s, dim=-1)              # [B, C]
    with torch.no_grad():
        t_lse = torch.logsumexp(t, dim=-1)          # [B, C]

    if select_topk_by == "student":
        _, topk_idx = s.topk(top_k, dim=-1)         # [B, C, K]
    else:
        with torch.no_grad():
            _, topk_idx = t.topk(top_k, dim=-1)     # [B, C, K]

    s_lp = s.gather(-1, topk_idx) - rearrange(s_lse, 'b c -> b c 1')  # [B, C, K]
    with torch.no_grad():
        t_lp = t.gather(-1, topk_idx) - rearrange(t_lse, 'b c -> b c 1')  # [B, C, K]

    return s_lp, t_lp, topk_idx


def compute_topk_logprobs_for_distillation(
    student_logits: torch.Tensor,   # [B, T, V]
    teacher_logits: torch.Tensor,   # [B, T, V]
    top_k: int = 100,
    select_topk_by: Literal["student", "teacher"] = "student",
    student_chunk_size: int = -1,
    teacher_chunk_size: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute top-K log-probabilities for distillation (non-distributed).

    Returns:
        s_topk_logprobs: [B, T, K]
        t_topk_logprobs: [B, T, K]
        topk_idx:        [B, T, K]
    """
    T = student_logits.shape[1]
    chunk = _effective_chunk(T, student_chunk_size, teacher_chunk_size)
    s_parts, t_parts, idx_parts = [], [], []
    for t0, t1 in _chunk_range(T, chunk):
        s_lp, t_lp, idx = _topk_logprobs_slice(
            student_logits[:, t0:t1],   # [B, C, V]
            teacher_logits[:, t0:t1],   # [B, C, V]
            top_k, select_topk_by,
        )
        s_parts.append(s_lp)    # [B, C, K]
        t_parts.append(t_lp)    # [B, C, K]
        idx_parts.append(idx)   # [B, C, K]
    return (
        torch.cat(s_parts, dim=1),    # [B, T, K]
        torch.cat(t_parts, dim=1),    # [B, T, K]
        torch.cat(idx_parts, dim=1),  # [B, T, K]
    )
