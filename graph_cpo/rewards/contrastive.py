"""Contrastive relevance reward:

R_rel(q_t, v_t) = cos(Enc(q_t), h_{v_t}) - eta * Delta_t
Delta_t = max_{u in V_res} cos(Enc(q_t), h^(L)_u)   (0 if V_res empty)

With rewards.rel_space == 'text' (default/dev), h is realized as Enc(d(v)) —
the same frozen encoder applied to the node's verbalization, keeping question
and node embeddings in one space. With 'gnn_projected', GNN embeddings are
mapped into the encoder space by a learned projector.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch

from ..models.encoder import SentenceEncoder
from ..planning.action_frame import verbalize_node


class ContrastiveRelevanceReward:
    def __init__(self, encoder: SentenceEncoder, eta: float = 0.5,
                 node_emb: Optional[Dict[str, torch.Tensor]] = None):
        self.encoder = encoder
        self.eta = eta
        self.node_emb = node_emb or {}

    def set_node_embeddings(self, node_emb: Dict[str, torch.Tensor]) -> None:
        self.node_emb = node_emb

    def embed_node_text(self, sdg, node_id: str) -> torch.Tensor:
        """Encode the verbalized node in the shared encoder space."""
        if node_id in self.node_emb:
            return self.node_emb[node_id]
        node = sdg.node(node_id)
        emb = self.encoder.encode(verbalize_node(node))
        self.node_emb[node_id] = emb
        return emb

    def __call__(self, question: str, sdg, target_node: str,
                 resolved: Iterable[str]) -> float:
        e_q = self.encoder.encode(question)
        h_v = self.embed_node_text(sdg, target_node)
        rel = SentenceEncoder.cosine(e_q, h_v)
        resolved = list(resolved)
        delta = 0.0
        if resolved:
            delta = max(SentenceEncoder.cosine(e_q, self.embed_node_text(sdg, u))
                        for u in resolved if u in sdg.node_ids())
        return rel - self.eta * delta
