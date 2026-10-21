"""Biaffine relation classifier (stage 2).

Predicts one edge type from {quantity-of, op-input, derive, condition,
prerequisite, goal-dep, null} for each candidate node pair. Candidate pairs are
formed when nodes occur in the same sentence, share an entity/unit, are linked
by an arithmetic cue, or connect to the goal.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .schema import EDGE_TYPES, SDG, V_Q_TYPES

LABELS = EDGE_TYPES + ["null"]
L2I = {l: i for i, l in enumerate(LABELS)}


class BiaffineClassifier(nn.Module):
    def __init__(self, hidden: int, num_labels: int = len(LABELS)):
        super().__init__()
        self.src_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.dst_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        # biaffine: [src, 1]^T W [dst, 1]
        self.W = nn.Parameter(torch.randn(num_labels, hidden + 1, hidden + 1) * 0.01)

    def forward(self, h_src: torch.Tensor, h_dst: torch.Tensor) -> torch.Tensor:
        """h_src/h_dst: (B, H) node representations -> (B, num_labels) scores."""
        s = torch.cat([self.src_mlp(h_src), torch.ones_like(h_src[:, :1])], dim=-1)  # (B, H+1)
        d = torch.cat([self.dst_mlp(h_dst), torch.ones_like(h_dst[:, :1])], dim=-1)  # (B, H+1)
        # (B, L) = s^T W_l d
        sd = torch.einsum("bi,lij->blj", s, self.W)          # (B, L, H+1)
        return torch.einsum("blj,bj->bl", sd, d)


def candidate_pairs(sdg: SDG) -> List[Tuple[str, str]]:
    """Node pairs worth classifying (candidate-pair heuristics)."""
    pairs = set()
    nodes = sdg.nodes
    goal_ids = {n.id for n in sdg.goal_nodes()}
    sentences = _split_sentences(sdg.text)

    def same_sentence(a, b) -> bool:
        return any(a.start >= s0 and a.end <= s1 and b.start >= s0 and b.end <= s1
                   for s0, s1 in sentences) if (a.start >= 0 and b.start >= 0) else True

    for i, a in enumerate(nodes):
        for b in nodes:
            if a.id == b.id:
                continue
            if a.type not in V_Q_TYPES and b.type not in V_Q_TYPES:
                continue
            if same_sentence(a, b) or (a.unit and a.unit == b.unit) \
                    or b.id in goal_ids or a.id in goal_ids:
                pairs.add((a.id, b.id))
    return sorted(pairs)


def _split_sentences(text: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"[^.!?]+[.!?]?", text) if m.group().strip()]


@torch.no_grad()
def predict_edges(classifier: BiaffineClassifier, node_repr: dict, sdg: SDG,
                  min_conf: float = 0.3, device="cpu") -> List:
    """Predict typed edges for a graph given per-node representations.

    node_repr: {node_id: FloatTensor[H]}
    """
    from .schema import SDGEdge
    pairs = candidate_pairs(sdg)
    if not pairs:
        return []
    h_src = torch.stack([node_repr[a] for a, _ in pairs]).to(device)
    h_dst = torch.stack([node_repr[b] for _, b in pairs]).to(device)
    scores = classifier(h_src, h_dst)
    probs = torch.softmax(scores, dim=-1).cpu()
    null_idx = L2I["null"]
    edges = []
    for (a, b), p in zip(pairs, probs):
        lab = int(p.argmax())
        if lab == null_idx or float(p[lab]) < min_conf:
            continue
        edges.append(SDGEdge(src=a, dst=b, type=LABELS[lab], conf=float(p[lab])))
    return edges


def train_features_from_silver(sdg: SDG) -> Tuple[dict, List[Tuple[str, str, str]]]:
    """(node text features, gold (src, dst, type) triples) for one silver graph."""
    feats = {n.id: {"text": n.text, "type": n.type, "num": n.num_value} for n in sdg.nodes}
    gold = [(e.src, e.dst, e.type) for e in sdg.edges]
    return feats, gold
