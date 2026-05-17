"""Einsum-annotated rewrite of DPOLoss from alignment_losses.py.

Shape key:
  B  — batch size
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, reduce


class DPOLoss(nn.Module):
    """Direct Preference Optimization loss (Rafailov et al., 2023).

    The core DPO loss maximizes the margin between chosen and rejected responses
    in log-probability space, normalized by a reference model.

    Loss = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))

    where log_ratio = log(pi(y|x) / pi_ref(y|x))

    See Chapter 8 for derivation.
    """

    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0):
        """
        Args:
            beta: Temperature parameter controlling KL penalty strength.
                  Higher beta = stronger preference signal, risk of overfitting.
                  Lower beta = more regularization toward reference model.
                  Typical values: 0.1-0.5
            label_smoothing: For cDPO variant. Assumes this fraction of labels
                            are incorrect. 0.0 = standard DPO, 0.1 = 10% noise.
        """
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,    # [B]
        policy_rejected_logps: torch.Tensor,  # [B]
        ref_chosen_logps: torch.Tensor,       # [B]
        ref_rejected_logps: torch.Tensor,     # [B]
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            policy_chosen_logps: Log probs of chosen responses from policy (batch,)
            policy_rejected_logps: Log probs of rejected responses from policy (batch,)
            ref_chosen_logps: Log probs of chosen responses from reference (batch,)
            ref_rejected_logps: Log probs of rejected responses from reference (batch,)

        Returns:
            loss: Scalar loss value
            metrics: Dict with chosen_rewards, rejected_rewards, margins
        """
        # Compute log ratios (implicit rewards)        [B]
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        # DPO logits: difference in log ratios          [B]
        dpo_logits = chosen_logratios - rejected_logratios

        # cDPO: label smoothing for noisy preferences
        if self.label_smoothing > 0:
            losses = (                                  # [B]
                -F.logsigmoid(self.beta * dpo_logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * dpo_logits) * self.label_smoothing
            )
        else:
            losses = -F.logsigmoid(self.beta * dpo_logits)  # [B]

        # Compute implicit rewards for logging          [B]
        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()

        # Scalar mean via einsum: [B] -> scalar
        loss = reduce(losses, 'b -> ', 'mean')

        margins = chosen_rewards - rejected_rewards     # [B]
        metrics = {
            "chosen_rewards": reduce(chosen_rewards, 'b -> ', 'mean').item(),
            "rejected_rewards": reduce(rejected_rewards, 'b -> ', 'mean').item(),
            "margins": reduce(margins, 'b -> ', 'mean').item(),
            "accuracy": (chosen_rewards > rejected_rewards).float().mean().item(),
        }

        return loss, metrics
