"""Scalar-GRPO baseline (the "GRPO scalar reward" ablation variant).

Identical rollout/reward machinery as Graph-CPO, but the three reward dimensions
are scalarized BEFORE group normalization (the recipe Graph-CPO argues against).
Used for the reward-collapse comparison (training-curve runs) and the
"GRPO scalar reward" row of the reward-ablation table.
"""
from __future__ import annotations

import dataclasses

from ..config import RLConfig
from .rl import GraphCPOTrainer


def make_scalar_grpo_trainer(policy, ref_policy, engine, bundle, cfg: RLConfig,
                             planning_cfg, **kwargs) -> GraphCPOTrainer:
    scalar_cfg = dataclasses.replace(cfg, mode="scalar")
    return GraphCPOTrainer(policy, ref_policy, engine, bundle, scalar_cfg,
                           planning_cfg, **kwargs)


__all__ = ["make_scalar_grpo_trainer"]
