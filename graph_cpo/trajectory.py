"""Sequential decision loop over the Semantic Dependency Graph.

At each timestep: select a frontier node -> build the pedagogical action frame
-> generate one question conditioned on the serialized frame + graph prefix ->
apply the resolution gate -> update the graph state. This produces the
step-by-step generation trace of the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from .config import PlanningConfig
from .models.policy import build_prompt
from .planning.action_frame import build_action_frame, serialize_frame
from .planning.frontier import FrontierTracker
from .planning.prefix import GraphPrefixProjector
from .planning.selector import select_frontier_node
from .rewards.bundle import QuestionReward
from .rewards.format import extract_question
from .sdg.schema import SDG


@dataclass
class Timestep:
    target_node: str
    frame_text: str
    question: str                       # raw generation (may contain tags)
    node_emb: Optional[torch.Tensor] = None   # h^(L)_{v_t} for the graph prefix
    graph_prefix: Optional[torch.Tensor] = None   # p_t = W_p[h_v ⊕ Pool(N_1(v))]
    rewards: Optional[QuestionReward] = None
    resolved: bool = False


@dataclass
class Trajectory:
    problem_id: str
    problem_text: str = ""
    timesteps: List[Timestep] = field(default_factory=list)

    @property
    def questions(self) -> List[str]:
        return [extract_question(ts.question) for ts in self.timesteps]


def _tracker(sdg: SDG, gate: Optional[Callable[[str, str], bool]]) -> FrontierTracker:
    return FrontierTracker(sdg, resolve_gate=gate)


def select_and_frame(sdg: SDG, tracker: FrontierTracker, cfg: PlanningConfig,
                     node_emb: Optional[Dict[str, torch.Tensor]] = None,
                     prefix_projector=None) -> Optional[Timestep]:
    """Frontier selection + action-frame serialization for one state."""
    frontier = tracker.frontier()
    if not frontier:
        return None
    node_id = select_frontier_node(
        sdg, frontier, sorted(tracker.state.resolved), node_emb=node_emb,
        lambda_g=cfg.lambda_g, lambda_d=cfg.lambda_d,
        lambda_c=cfg.lambda_c, lambda_r=cfg.lambda_r)
    frame = build_action_frame(sdg, node_id, sorted(tracker.state.resolved))
    graph_prefix = None
    if prefix_projector is not None and node_emb and node_id in node_emb:
        nbr_embs = None
        nbrs = [node_emb[n] for n in sdg.one_hop(node_id) if n in node_emb]
        if nbrs:
            nbr_embs = torch.stack(nbrs)
        # detached: the projector is trained in the SFT phase only; the policy
        # has its own trainable prefix_proj applied inside the model
        graph_prefix = prefix_projector(node_emb[node_id], nbr_embs).detach()
    return Timestep(target_node=node_id, frame_text=serialize_frame(frame),
                    question="", node_emb=node_emb.get(node_id) if node_emb else None,
                    graph_prefix=graph_prefix)


def sample_trajectories(engine, sdg: SDG, cfg: PlanningConfig,
                        node_emb: Optional[Dict[str, torch.Tensor]] = None,
                        gate: Optional[Callable[[str, str], bool]] = None,
                        num_rollouts: int = 8, max_new_tokens: int = 96,
                        prefix_projector=None) -> List[Trajectory]:
    """Sample M full trajectories; each rollout advances its own graph state.

    Prompts are batched per timestep across rollouts (all M prompts at once).
    Returns M Trajectories with per-timestep target frames and questions.
    """
    trajs = [_TrajectoryRunner(sdg, cfg, node_emb, gate, prefix_projector)
             for _ in range(num_rollouts)]
    for r in trajs:
        r.trajectory.problem_text = sdg.text
    for _ in range(cfg.max_questions):
        active = [r for r in trajs if r.pending_frame() is not None]
        if not active:
            break
        prompts, prefixes = [], []
        for r in active:
            ts = r.pending_frame()
            prompts.append(build_prompt(sdg.text, r.questions(), ts.frame_text))
            prefixes.append(ts.graph_prefix)
        # fallback for timesteps without a graph prefix: same device/dtype.
        # If NO rollout has a prefix, pass None so the policy generates
        # unconditioned instead of feeding a degenerate tensor through
        # the prefix projection.
        ref = next((p for p in prefixes if p is not None), None)
        if ref is None:
            prefix_tensor = None
        else:
            prefixes = [p if p is not None else torch.zeros_like(ref)
                        for p in prefixes]
            prefix_tensor = torch.cat(
                [p if p.dim() == 2 else p.unsqueeze(0) for p in prefixes], dim=0)
        batches = engine.sample(prompts, graph_prefixes=prefix_tensor,
                                num_return=1, max_new_tokens=max_new_tokens)
        for r, batch in zip(active, batches):
            r.commit(batch[0])
    return [r.trajectory for r in trajs]


class _TrajectoryRunner:
    """One rollout: its own frontier state and question history."""

    def __init__(self, sdg: SDG, cfg: PlanningConfig,
                 node_emb: Optional[Dict[str, torch.Tensor]],
                 gate: Optional[Callable[[str, str], bool]],
                 prefix_projector=None):
        self.sdg = sdg
        self.cfg = cfg
        self.node_emb = node_emb or {}
        self.prefix_projector = prefix_projector
        self.tracker = _tracker(sdg, gate)
        self.trajectory = Trajectory(problem_id=sdg.problem_id)
        self._pending: Optional[Timestep] = None

    def questions(self) -> List[str]:
        return self.trajectory.questions

    def pending_frame(self) -> Optional[Timestep]:
        if self._pending is None and len(self.trajectory.timesteps) < self.cfg.max_questions:
            self._pending = select_and_frame(self.sdg, self.tracker, self.cfg,
                                             self.node_emb, self.prefix_projector)
        return self._pending

    def commit(self, question: str) -> None:
        ts = self._pending
        self._pending = None
        if ts is None:
            return
        ts.question = question
        ts.resolved = self.tracker.step(question, ts.target_node)
        self.trajectory.timesteps.append(ts)


def rollout_with_teacher(sdg: SDG, teacher, cfg: PlanningConfig,
                         gate: Optional[Callable[[str, str], bool]] = None
                         ) -> Trajectory:
    """Teacher-forced trajectory: align teacher steps to graph targets (SFT data)."""
    traj = Trajectory(problem_id=sdg.problem_id, problem_text=sdg.text)
    tracker = _tracker(sdg, gate)
    for step in teacher.steps:
        if step.target_node not in sdg.node_ids():
            continue
        frame = build_action_frame(sdg, step.target_node, sorted(tracker.state.resolved))
        traj.timesteps.append(Timestep(
            target_node=step.target_node, frame_text=serialize_frame(frame),
            question=step.question, resolved=True))
        tracker.force_resolve(step.target_node)
    return traj
