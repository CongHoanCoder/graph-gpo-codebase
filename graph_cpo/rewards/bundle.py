"""Bundle of the three pedagogical reward dimensions:

  R_rel  contrastive relevance  (targetedness / non-redundancy)
  R_coh  logical coherence      (sequential progression)
  R_form format adherence       (generation validity)

Also builds the node-resolution gate combining the similarity test
cos(Enc(q), h_v) >= tau_sim with the NLI test P(entail) >= tau_nli.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import torch

from ..models.encoder import SentenceEncoder
from .contrastive import ContrastiveRelevanceReward
from .coherence import NLICoherenceReward
from .format import FormatReward, extract_question

REWARD_DIMS = ["rel", "coh", "form"]


@dataclass
class QuestionReward:
    """Per-timestep rewards for one generated question (keyed by dimension)."""
    rel: float
    coh: float
    form: float

    def as_dict(self) -> Dict[str, float]:
        return {"rel": self.rel, "coh": self.coh, "form": self.form}


class RewardBundle:
    def __init__(self, encoder: SentenceEncoder, nli: NLICoherenceReward,
                 eta: float = 0.5, sim_threshold: float = 0.75,
                 nli_threshold: float = 0.6, strict_format: bool = True):
        self.rel = ContrastiveRelevanceReward(encoder, eta=eta)
        self.coh = nli
        self.form = FormatReward(strict=strict_format)
        self.encoder = encoder
        self.sim_threshold = sim_threshold
        self.nli_threshold = nli_threshold

    def __call__(self, question: str, sdg, target_node: str,
                 resolved: Iterable[str]) -> QuestionReward:
        q = extract_question(question)
        resolved = list(resolved)
        return QuestionReward(
            rel=self.rel(q, sdg, target_node, resolved),
            coh=self.coh(q, sdg, target_node),
            form=self.form(question),   # raw output, tags included
        )

    # -------------------------------------------------- resolution gate (Sec 3.1)
    def resolve_gate(self, question: str, sdg, node_id: str) -> bool:
        """True iff the question both aligns with the node and addresses it."""
        q = extract_question(question)
        h_q = self.encoder.encode(q)
        h_v = self.rel.embed_node_text(sdg, node_id)
        if SentenceEncoder.cosine(h_q, h_v) < self.sim_threshold:
            return False
        return self.coh.passes_gate(q, sdg, node_id, tau_nli=self.nli_threshold)

    def make_gate(self, sdg) -> Callable[[str, str], bool]:
        """Closure usable by FrontierTracker(step=...)."""
        return lambda question, node_id: self.resolve_gate(question, sdg, node_id)


def build_reward_bundle(rewards_cfg, planning_cfg, device: str = "cpu") -> RewardBundle:
    encoder = SentenceEncoder(rewards_cfg.encoder, device=device)
    nli = NLICoherenceReward(rewards_cfg.nli_model, device=device)
    return RewardBundle(encoder, nli, eta=rewards_cfg.eta,
                        sim_threshold=planning_cfg.sim_threshold,
                        nli_threshold=planning_cfg.nli_threshold)
