
import torch
from einops import einsum, reduce


def _masked_token_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # values: [B, T]   mask: [B, T]
    return einsum(values, mask, 'b t, b t -> ') / mask.sum().clamp(min=1)


def compute_jsd_loss(
    student_logprobs: torch.Tensor,   # [B, T, K]
    teacher_logprobs: torch.Tensor,   # [B, T, K]
    response_mask: torch.Tensor,      # [B, T]
    jsd_alpha: float = 0.5,
    renormalize: bool = True,
) -> torch.Tensor:                    # scalar
    """JSD(student || teacher) with mixture weight jsd_alpha (top-K selected by student).

    jsd_alpha=0.5 → symmetric JSD (SDPO default).
    jsd_alpha=0.0 → forward KL.  jsd_alpha=1.0 → reverse KL.
    """
    student_probs = student_logprobs.exp()                                   # [B, T, K]
    teacher_probs = teacher_logprobs.exp()                                   # [B, T, K]
    if renormalize:
        student_probs = student_probs / reduce(
            student_probs, 'b t k -> b t 1', 'sum'
        ).clamp(min=1e-8)                                                    # [B, T, K]
        teacher_probs = teacher_probs / reduce(
            teacher_probs, 'b t k -> b t 1', 'sum'
        ).clamp(min=1e-8)                                                    # [B, T, K]

    M = jsd_alpha * student_probs + (1.0 - jsd_alpha) * teacher_probs       # [B, T, K]
    log_M = M.clamp(min=1e-8).log()                                          # [B, T, K]

    kl_s = einsum(student_probs, student_logprobs - log_M, 'b t k, b t k -> b t')   # [B, T]
    kl_t = einsum(teacher_probs, teacher_logprobs - log_M, 'b t k, b t k -> b t')   # [B, T]

    per_token_jsd = jsd_alpha * kl_s + (1.0 - jsd_alpha) * kl_t            # [B, T]
    return _masked_token_mean(per_token_jsd, response_mask)                  # scalar
