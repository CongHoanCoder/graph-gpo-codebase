"""Targeted supervised warm start: imitate teacher Socratic sequences.

The Graph-Augmented Policy imitates teacher Socratic sequences conditioned on
the problem, prior questions, serialized action frame, and graph prefix. This
produces the warm-started policy pi_theta (and the KL reference pi_ref) that
mitigate cold-start instability in the RL phase.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch

from ..config import SFTConfig
from ..models.policy import build_prompt
from ..planning.prefix import GraphPrefixProjector
from ..rewards.format import extract_question
from ..trajectory import Trajectory
from .common import nll_of_answer, set_seed


class SFTTrainer:
    def __init__(self, policy, cfg: SFTConfig,
                 prefix_projector: Optional[GraphPrefixProjector] = None,
                 device: str = "cpu", log_fn=print):
        self.policy = policy.to(device)
        self.cfg = cfg
        self.prefix = prefix_projector
        self.device = device
        self.log = log_fn
        params = [p for p in policy.parameters() if p.requires_grad]
        if prefix_projector is not None:
            params += list(prefix_projector.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=cfg.lr)

    def _prefix_for(self, traj: Trajectory, t_idx: int,
                    node_emb: Optional[Dict[str, torch.Tensor]] = None
                    ) -> Optional[torch.Tensor]:
        if self.prefix is None or node_emb is None:
            return None
        ts = traj.timesteps[t_idx]
        if ts.target_node not in node_emb:
            return None
        return self.prefix(node_emb[ts.target_node], None)

    def train(self, trajectories: List[Trajectory],
              node_embeddings: Optional[Dict[str, Dict[str, torch.Tensor]]] = None) -> List[float]:
        """One SFT pass per epoch over teacher trajectories; returns epoch losses."""
        set_seed(0)
        losses: List[float] = []
        self.policy.train()
        for epoch in range(self.cfg.epochs):
            total, count = 0.0, 0
            step = 0
            self.optimizer.zero_grad()
            for i, traj in enumerate(trajectories):
                if not traj.timesteps:
                    continue
                embs = (node_embeddings or {}).get(traj.problem_id)
                history: List[str] = []
                for ts in traj.timesteps:
                    prompt = build_prompt(traj.problem_text, history, ts.frame_text)
                    prefix = None
                    if self.prefix is not None and embs and ts.target_node in embs:
                        prefix = self.prefix(embs[ts.target_node], None)
                    # ts.question carries the <question> tags; history shows
                    # the extracted text (same view as rollout-time prompts)
                    loss = nll_of_answer(self.policy, prompt, ts.question,
                                         graph_prefix=prefix)
                    (loss / self.cfg.grad_accum).backward()
                    total += float(loss.detach())
                    count += 1
                    history.append(extract_question(ts.question))
                    step += 1
                    if step % self.cfg.grad_accum == 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.policy.parameters() if p.requires_grad], 1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        if step <= self.cfg.warmup_steps:
                            self._warmup_lr(step)
            if count:
                self.optimizer.step()
                self.optimizer.zero_grad()
            avg = total / max(count, 1)
            losses.append(avg)
            self.log(f"[sft] epoch {epoch+1}/{self.cfg.epochs} loss={avg:.4f}")
        self.policy.eval()
        return losses

    def _warmup_lr(self, step: int) -> None:
        base = self.cfg.lr
        scale = min(1.0, step / max(self.cfg.warmup_steps, 1))
        for g in self.optimizer.param_groups:
            g["lr"] = base * scale

    def save(self, path: str | Path) -> None:
        self.policy.save_pretrained(str(path))
        self.log(f"[sft] checkpoint saved to {path}")
