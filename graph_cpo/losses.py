"""Graph-CPO objective: clipped surrogate + KL regularization + anchor blend.

    L_CPO = -E_{m,t}[ min( r_{m,t} * hat{A}_{m,t}, clip(r) * hat{A}_{m,t}) ]
            + beta * E_{m,t}[ KL( pi_theta || pi_ref ) ]

    L_Graph-CPO = alpha(u) * L_target + (1 - alpha(u)) * L_CPO

r_{m,t} = pi_theta(a|s) / pi_{theta_old}(a|s); the reference policy pi_ref is
used only in the KL term (it is the warm-start SFT checkpoint).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def clipped_surrogate(logp_new: torch.Tensor, logp_old: torch.Tensor,
                      advantage: torch.Tensor, mask: torch.Tensor,
                      clip_eps: float = 0.2) -> torch.Tensor:
    """Per-token PPO/GRPO clipped objective, masked mean over valid tokens.

    All tensors: (B, T) or broadcastable; mask is (B, T) with 1 on action tokens.
    """
    ratio = torch.exp(logp_new - logp_old)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    loss = -torch.min(unclipped, clipped)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def approx_kl(logp_new: torch.Tensor, logp_old: torch.Tensor,
              mask: torch.Tensor) -> torch.Tensor:
    """Unbiased KL estimator used for monitoring (Schulman blog): mean(ratio-1-log r)."""
    ratio = torch.exp(logp_new - logp_old)
    kl = ratio - 1.0 - (logp_new - logp_old)
    return (kl * mask).sum() / mask.sum().clamp(min=1.0)


def kl_to_reference(logp_new: torch.Tensor, logp_ref: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """D_KL(pi_theta || pi_ref) per-token masked mean (k3 estimator)."""
    log_ratio = logp_ref - logp_new                 # log(pi_ref/pi_theta)
    ratio = torch.exp(log_ratio)
    kl = (ratio - 1.0) - log_ratio                  # >= 0
    return (kl * mask).sum() / mask.sum().clamp(min=1.0)


def graph_cpo_loss(logp_new: torch.Tensor, logp_old: torch.Tensor,
                   logp_ref: Optional[torch.Tensor], advantage: torch.Tensor,
                   mask: torch.Tensor, clip_eps: float = 0.2,
                   kl_beta: float = 0.01) -> Dict[str, torch.Tensor]:
    """Full L_CPO with optional reference-KL penalty. Returns loss + diagnostics."""
    surrogate = clipped_surrogate(logp_new, logp_old, advantage, mask, clip_eps)
    out: Dict[str, torch.Tensor] = {
        "surrogate": surrogate,
        "kl_old": approx_kl(logp_new, logp_old, mask),
        "clip_frac": ((torch.exp(logp_new - logp_old) - 1.0).abs()
                      > clip_eps).float().mul(mask).sum() / mask.sum().clamp(min=1.0),
    }
    total = surrogate
    if logp_ref is not None and kl_beta > 0:
        kl_ref = kl_to_reference(logp_new, logp_ref, mask)
        out["kl_ref"] = kl_ref
        total = total + kl_beta * kl_ref
    out["loss"] = total
    return out


def total_loss(l_target: Optional[torch.Tensor], l_cpo: torch.Tensor,
               alpha: float) -> torch.Tensor:
    """L = alpha(u) * L_target + (1 - alpha(u)) * L_CPO."""
    if l_target is None:
        return l_cpo
    return alpha * l_target + (1.0 - alpha) * l_cpo
