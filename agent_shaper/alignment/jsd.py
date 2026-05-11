
import torch

def _masked_token_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp(min=1)
    
def compute_jsd_loss(
    student_logprobs: torch.Tensor,   
    teacher_logprobs: torch.Tensor,   
    response_mask: torch.Tensor,      
    jsd_alpha: float = 0.5,
    renormalize: bool = True,
) -> torch.Tensor:
    """JSD(student || teacher) with mixture weight jsd_alpha (top-K selected by student).

    jsd_alpha=0.5 → symmetric JSD (SDPO default).
    jsd_alpha=0.0 → forward KL.  jsd_alpha=1.0 → reverse KL.
    """
    student_probs = student_logprobs.exp()
    teacher_probs = teacher_logprobs.exp()
    if renormalize:
        student_probs = student_probs / student_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    M = jsd_alpha * student_probs + (1.0 - jsd_alpha) * teacher_probs
    log_M = M.clamp(min=1e-8).log()
    kl_s = (student_probs * (student_logprobs - log_M)).sum(dim=-1)
    kl_t = (teacher_probs * (teacher_logprobs - log_M)).sum(dim=-1)
    per_token_jsd = jsd_alpha * kl_s + (1.0 - jsd_alpha) * kl_t
    return _masked_token_mean(per_token_jsd, response_mask)