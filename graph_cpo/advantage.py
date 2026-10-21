"""Decoupled adaptive advantage + curriculum weighting.

For each prompt we sample M rollouts. Per reward dimension k:

    A_{m,k,t} = (R_{m,k,t} - mu_{k,t}) / (sigma_{k,t} + eps)          (group-normalized
                                                                      PER DIMENSION,
                                                                      before any
                                                                      aggregation)

    hat{A}_{m,t} = Norm_m( sum_k omega_k(u) * A_{m,k,t} )

Curriculum:
    omega_form(u) = omega_min + (omega_max - omega_min) * exp(-lambda * u)
    omega_rel(u)  = omega_coh(u) = (1 - omega_form(u)) / 2

This is the mechanism that prevents the easy format objective from dominating
the harder graph-grounded signals ("reward collapse"): a saturated dimension
has sigma_k -> 0, and the group normalization yields zero advantage for it, so
it cannot drown out relevance/coherence even while its raw reward is near 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .rewards.bundle import REWARD_DIMS


def curriculum_weights(step: int, omega_min: float = 0.1, omega_max: float = 0.8,
                       lam: float = 0.01) -> Dict[str, float]:
    w_form = omega_min + (omega_max - omega_min) * float(np.exp(-lam * step))
    w_rest = (1.0 - w_form) / 2.0
    return {"form": w_form, "rel": w_rest, "coh": w_rest}


def anchor_alpha(step: int, alpha0: float = 0.8, alpha_min: float = 0.05,
                 lam: float = 0.01) -> float:
    """alpha(u): decay of the supervised anchoring term."""
    return alpha_min + (alpha0 - alpha_min) * float(np.exp(-lam * step))


def _group_normalize(values: np.ndarray, eps: float) -> np.ndarray:
    """Per-dimension GRPO-style normalization over the M rollouts of one prompt.

    values: (M, T) rewards for one dimension (NaN = padded slot); returns
    advantages (M, T). Degenerate (near-zero) variance -> zero advantage, so a
    saturated reward contributes nothing (e.g. all rollouts format-perfect =>
    A_form = 0 — the mechanism that stops easy objectives dominating).
    """
    nan_mask = np.isnan(values)
    filled = np.where(nan_mask, 0.0, values)
    counts = np.where(nan_mask, 0.0, 1.0).sum(axis=0, keepdims=True)
    mu = np.where(counts > 0, filled.sum(axis=0, keepdims=True) / np.maximum(counts, 1.0), 0.0)
    var = np.where(counts > 0,
                   ((filled - mu) ** 2 * ~nan_mask).sum(axis=0, keepdims=True)
                   / np.maximum(counts, 1.0), 0.0)
    sigma = np.sqrt(var)
    out = (filled - mu) / (sigma + eps)
    return np.where(nan_mask, 0.0, out)


@dataclass
class AdvantageResult:
    advantage: np.ndarray          # (M, T) composite hat{A}_{m,t}
    per_dim: Dict[str, np.ndarray] # (M, T) group-normalized A_{m,k,t}
    weights: Dict[str, float]      # curriculum weights at this step


def compute_advantage(rewards: Dict[str, np.ndarray], step: int,
                      omega_min: float = 0.1, omega_max: float = 0.8,
                      lam: float = 0.01, eps: float = 1.0e-8,
                      normalize_composite: bool = False,
                      use_dims: Optional[Sequence[str]] = None,
                      dynamic_weighting: bool = True) -> AdvantageResult:
    """Composite decoupled advantage for a single prompt.

    rewards: {"rel": (M,T), "coh": (M,T), "form": (M,T)} per-timestep rewards
             of M rollouts; NaN marks padded slots (masked out of both the
             normalization statistics and the loss).

    The per-dimension group normalization A_{m,k,t} is the normalization step; the
    curriculum-weighted sum is used directly as hat{A} (matching the worked
    worked example, 0.42*0.86 + 0.42*0.60 + 0.16*0 = 0.61). Set
    normalize_composite=True to additionally renormalize the combined value.
    """
    w = curriculum_weights(step, omega_min, omega_max, lam)
    if not dynamic_weighting:
        # "w/o dynamic weighting" ablation: fixed uniform-ish weights
        w = {k: 1.0 / len(REWARD_DIMS) for k in REWARD_DIMS}
    if use_dims is None:
        use_dims = REWARD_DIMS
    per_dim: Dict[str, np.ndarray] = {}
    composite = np.zeros_like(np.asarray(rewards["rel"], dtype=np.float64))
    for k in REWARD_DIMS:
        a_k = _group_normalize(np.asarray(rewards[k], dtype=np.float64), eps)
        per_dim[k] = a_k
        if k in use_dims:
            composite = composite + w[k] * a_k
    if normalize_composite:
        composite = _group_normalize(composite, eps)
    return AdvantageResult(advantage=composite, per_dim=per_dim, weights=w)


def scalar_grpo_advantage(rewards: Dict[str, np.ndarray], step: int,
                          weights: Sequence[float] = (1.0, 1.0, 1.0),
                          eps: float = 1.0e-8) -> np.ndarray:
    """Ablation baseline: scalarize FIRST, then group-normalize (the GRPO
    recipe — easy objectives mask harder ones)."""
    scalar = np.zeros_like(np.asarray(rewards["rel"], dtype=np.float64))
    nan_all = np.isnan(np.asarray(rewards["rel"], dtype=np.float64))
    for k, wgt in zip(REWARD_DIMS, weights):
        vals = np.asarray(rewards[k], dtype=np.float64)
        scalar = scalar + wgt * np.where(np.isnan(vals), 0.0, vals)
    scalar[nan_all] = np.nan
    return _group_normalize(scalar, eps)
