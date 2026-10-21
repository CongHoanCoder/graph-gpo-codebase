"""Graph-CPO reinforcement learning trainer.

Per training update:
  1. Roll out M trajectories from pi_{theta_old} through the SDG decision
     process (frontier selection -> action frame -> question -> resolution gate).
  2. Score each timestep with the three reward dimensions (relevance,
     coherence, format).
  3. Decoupled adaptive advantage: per-dimension group normalization, then
     curriculum-weighted combination (Eqs. 12-13) — or, in "scalar" mode, the
     GRPO baseline that scalarizes before normalization (ablation).
  4. Clipped CPO loss with KL to the frozen reference pi_ref,
     blended with the supervised anchoring loss L_target via alpha(u).

Per-dimension raw reward means are logged every update, which produces the
reward-collapse training curves (contrastive relevance, format adherence, and
logical coherence vs. update step).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from ..advantage import anchor_alpha, compute_advantage, curriculum_weights, scalar_grpo_advantage
from ..config import RLConfig
from ..losses import graph_cpo_loss, total_loss
from ..rewards.bundle import REWARD_DIMS, RewardBundle
from ..rewards.format import extract_question
from ..sdg.schema import SDG
from ..trajectory import Trajectory, sample_trajectories
from .common import gen_token_logprobs, nll_of_answer, pad_trajectory_rewards, set_seed


@dataclass
class UpdateLog:
    step: int
    loss: float
    surrogate: float
    kl_old: float
    kl_ref: Optional[float]
    clip_frac: float
    alpha: float
    omega: Dict[str, float]
    reward_means: Dict[str, float]     # RAW per-dimension reward means (Fig. curves)


class GraphCPOTrainer:
    def __init__(self, policy, ref_policy, engine, bundle: RewardBundle,
                 cfg: RLConfig, planning_cfg, prefix_projector=None,
                 teacher_by_problem: Optional[Dict[str, Callable]] = None,
                 node_embeddings: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
                 device: str = "cpu", log_fn=print):
        self.policy = policy.to(device)
        self.ref_policy = ref_policy.to(device) if ref_policy is not None else None
        self.engine = engine
        self.bundle = bundle
        self.cfg = cfg
        self.planning_cfg = planning_cfg
        self.prefix = prefix_projector
        self.teachers = teacher_by_problem or {}
        self.node_emb = node_embeddings or {}
        self.device = device
        self.log = log_fn
        self.optimizer = torch.optim.AdamW(
            [p for p in policy.parameters() if p.requires_grad], lr=cfg.lr)
        self.global_step = 0
        self.history: List[UpdateLog] = []

    # ------------------------------------------------------------------ data
    def _score(self, traj: Trajectory, sdg: SDG, gate) -> None:
        """Attach per-timestep rewards to a trajectory."""
        for ts in traj.timesteps:
            resolved = [u for u in traj_timesteps_resolved(traj, ts)]
            ts.rewards = self.bundle(ts.question, sdg, ts.target_node, resolved)
            ts.resolved = gate(ts.question, ts.target_node)

    # ------------------------------------------------------------- one update
    def update_on_problem(self, sdg: SDG) -> Optional[UpdateLog]:
        gate = self.bundle.make_gate(sdg)
        embs = self.node_emb.get(sdg.problem_id)
        self.policy.eval()
        trajs = sample_trajectories(
            self.engine, sdg, self.planning_cfg, node_emb=embs, gate=gate,
            num_rollouts=self.cfg.rollouts_per_prompt,
            max_new_tokens=64, prefix_projector=self.prefix)
        if not any(t.timesteps for t in trajs):
            return None
        for t in trajs:
            self._score(t, sdg, gate)

        rewards, _mask = pad_trajectory_rewards(trajs, list(REWARD_DIMS))

        u = self.global_step
        if self.cfg.mode == "scalar":
            adv_np = scalar_grpo_advantage(rewards, u)
        else:
            adv = compute_advantage(
                rewards, step=u,
                omega_min=self.cfg.omega_min, omega_max=self.cfg.omega_max,
                lam=self.cfg.curriculum_lambda, eps=self.cfg.adv_eps,
                use_dims=self.cfg.use_rewards,
                dynamic_weighting=self.cfg.dynamic_weighting)
            adv_np = adv.advantage
        advantage = torch.tensor(np.nan_to_num(adv_np), dtype=torch.float32,
                                 device=self.device)
        weights = (curriculum_weights(u, self.cfg.omega_min, self.cfg.omega_max,
                                      self.cfg.curriculum_lambda)
                   if self.cfg.dynamic_weighting else
                   {k: 1.0 / len(REWARD_DIMS) for k in REWARD_DIMS})

        # --- pass 1 (no grad): collect valid timesteps with behavior
        # (theta_old) and reference logprobs. Tokenization-only emptiness
        # checks decide validity, so this pass fixes the item count N.
        items: List[dict] = []
        for m, traj in enumerate(trajs):
            if not traj.timesteps:
                continue
            history: List[str] = []
            for t, ts in enumerate(traj.timesteps):
                prompt = _prompt_for(traj, history, ts.frame_text)
                prefix = ts.graph_prefix   # the exact prefix used at rollout time
                logp_old, _ = gen_token_logprobs(self.policy, prompt, ts.question, prefix)
                if logp_old.numel() == 0:
                    history.append(extract_question(ts.question))
                    continue
                items.append(dict(
                    m=m, t=t, prompt=prompt, prefix=prefix,
                    question=ts.question, target_node=ts.target_node,
                    history=list(history),
                    logp_old=logp_old,
                    logp_ref=self._ref_logprobs(prompt, ts.question, prefix)))
                # prompts must match rollout-time construction, which shows
                # the extracted (untagged) history
                history.append(extract_question(ts.question))

        if not items:
            return None

        # --- pass 2 (with grad): one grad-enabled forward per item, backward
        # immediately (scaled by 1/N), so only a single autograd graph is
        # alive at a time (memory-bounded micro-batching).
        a = anchor_alpha(u, self.cfg.alpha0, self.cfg.alpha_min,
                         self.cfg.alpha_lambda)
        n_items = len(items)
        self.policy.train()
        self.optimizer.zero_grad()
        surro, kls, clipfr, anchors = [], [], [], []
        total_val = 0.0
        for it in items:
            logp_new_t = self._current_logprobs(it["prompt"], it["question"],
                                                it["prefix"])
            if logp_new_t.numel() == 0:
                continue
            # the timestep advantage applies to every action token of q_t
            adv_t = advantage[it["m"], it["t"]].repeat(
                logp_new_t.numel()).unsqueeze(0)
            msk = torch.ones_like(adv_t)
            out = graph_cpo_loss(logp_new_t.unsqueeze(0), it["logp_old"].unsqueeze(0),
                                 it["logp_ref"], adv_t, msk,
                                 clip_eps=self.cfg.clip_eps,
                                 kl_beta=self.cfg.kl_beta)
            # supervised anchor L_target when a teacher trajectory exists
            anchor = self._anchor_loss(sdg.problem_id, it["prompt"],
                                       it["target_node"], it["history"],
                                       it["prefix"])
            loss = total_loss(anchor, out["loss"], a) / n_items
            loss.backward()          # frees this item's graph right away
            total_val += float(loss.detach())
            surro.append(float(out["surrogate"].detach()))
            kls.append(float(out["kl_old"].detach()))
            clipfr.append(float(out["clip_frac"].detach()))
            if anchor is not None:
                anchors.append(float(anchor.detach()))

        if not surro:
            return None
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.policy.parameters() if p.requires_grad], 1.0)
        self.optimizer.step()
        self.global_step += 1

        log = UpdateLog(
            step=self.global_step, loss=total_val,
            surrogate=float(np.mean(surro)) if surro else 0.0,
            kl_old=float(np.mean(kls)) if kls else 0.0,
            kl_ref=None, clip_frac=float(np.mean(clipfr)) if clipfr else 0.0,
            alpha=anchor_alpha(u, self.cfg.alpha0, self.cfg.alpha_min,
                               self.cfg.alpha_lambda),
            omega=weights,
            reward_means={k: _nanmean(rewards[k]) for k in REWARD_DIMS})
        self.history.append(log)
        self.log(f"[rl] step {log.step} loss={log.loss:.4f} "
                 f"rel={log.reward_means['rel']:.3f} "
                 f"coh={log.reward_means['coh']:.3f} "
                 f"form={log.reward_means['form']:.3f} "
                 f"omega_form={log.omega['form']:.2f} alpha={log.alpha:.2f}")
        return log

    # ---------------------------------------------------------------- helpers
    def _current_logprobs(self, prompt: str, answer: str,
                          prefix: Optional[torch.Tensor]) -> torch.Tensor:
        """Policy logprobs WITH gradient (the theta side of the ratio)."""
        lp, _ = gen_token_logprobs(self.policy, prompt, answer, prefix, grad=True)
        return lp

    def _ref_logprobs(self, prompt: str, answer: str,
                      prefix: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.ref_policy is None:
            return None
        self.ref_policy.eval()
        lp, _ = gen_token_logprobs(self.ref_policy, prompt, answer, prefix)
        return lp

    def _anchor_loss(self, problem_id: str, prompt: str, target_node: str,
                     history, prefix: Optional[torch.Tensor]
                     ) -> Optional[torch.Tensor]:
        teacher = self.teachers.get(problem_id)
        if teacher is None:
            return None
        for step in teacher.steps:
            if step.target_node == target_node:
                return nll_of_answer(self.policy, prompt, step.question,
                                     graph_prefix=prefix)
        return None

    def attach_planning_config(self, planning_cfg) -> None:
        """Backwards-compatible alias; planning config is set in the constructor."""
        self.planning_cfg = planning_cfg

    def plot_ready(self) -> Dict[str, List[float]]:
        """Per-dimension raw reward trajectories for the reward-collapse figure."""
        return {
            "step": [h.step for h in self.history],
            "rel": [h.reward_means["rel"] for h in self.history],
            "coh": [h.reward_means["coh"] for h in self.history],
            "form": [h.reward_means["form"] for h in self.history],
            "omega_form": [h.omega["form"] for h in self.history],
        }


def _prompt_for(traj: Trajectory, history: List[str], frame_text: str) -> str:
    from ..models.policy import build_prompt
    return build_prompt(traj.problem_text, history, frame_text)


def traj_timesteps_resolved(traj: Trajectory, current) -> List[str]:
    """Nodes resolved by timesteps strictly before `current`."""
    out = []
    for ts in traj.timesteps:
        if ts is current:
            break
        if ts.resolved:
            out.append(ts.target_node)
    return out


def _nanmean(a: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    return float(a.mean()) if a.size else 0.0
