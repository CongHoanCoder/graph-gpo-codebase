"""Relation-aware GNN encoder (stage 3).

h_i^(l+1) = sigma( W_0 h_i^(l) + sum_r sum_{j in N_r(i)} (1/c_{i,r}) W_r h_j^(l) )

Node init: BERT(span) ⊕ e_type ⊕ phi_num  (see init_from_features).
Implemented in plain PyTorch (no DGL/PyG dependency).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .schema import EDGE_TYPES, NODE_TYPES, SDG, node_type_id

NUM_FEATS = 3  # [has_num, normalized magnitude, is_goal]


def numeric_features(num: Optional[float]) -> torch.Tensor:
    if num is None:
        return torch.zeros(NUM_FEATS)
    mag = torch.tensor([min(abs(num) / 1000.0, 1.0), 1.0 if num >= 0 else 0.0, 1.0])
    return mag


class RelationGNN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 768, num_relations: int = len(EDGE_TYPES),
                 num_layers: int = 3, num_node_types: int = len(NODE_TYPES)):
        super().__init__()
        self.hidden = hidden
        self.type_emb = nn.Embedding(num_node_types, hidden)
        self.input_proj = nn.Linear(in_dim + NUM_FEATS + hidden, hidden)
        self.self_lin = nn.ModuleList()
        self.rel_lins = nn.ModuleList()
        self.layers = num_layers
        for _ in range(num_layers):
            self.self_lin.append(nn.Linear(hidden, hidden))
            self.rel_lins.append(nn.ModuleList(
                [nn.Linear(hidden, hidden, bias=False) for _ in range(num_relations)]))
        self.activation = nn.ReLU()

    def forward(self, node_init: torch.Tensor, node_type_ids: torch.Tensor,
                edge_index: torch.Tensor, edge_type_ids: torch.Tensor) -> torch.Tensor:
        """node_init: (N, in_dim+NUM_FEATS); edge_index: (2, E) [src, dst];
        edge_type_ids: (E,). Returns node embeddings (N, hidden)."""
        h = self.input_proj(torch.cat(
            [node_init, self.type_emb(node_type_ids)], dim=-1))
        src, dst = edge_index[0], edge_index[1]
        for l in range(self.layers):
            msg = torch.zeros_like(h)
            cnt = torch.zeros(h.size(0), 1, device=h.device)
            for r in range(len(self.rel_lins[l])):
                mask = edge_type_ids == r
                if mask.any():
                    s = src[mask]
                    d = dst[mask]
                    msg.index_add_(0, d, self.rel_lins[l][r](h[s]))
                    cnt.index_add_(0, d, torch.ones(d.size(0), 1, device=h.device))
            h = self.activation(self.self_lin[l](h) + msg / cnt.clamp(min=1.0))
        return h


class GraphEncoderBundle(nn.Module):
    """Text-span encoder (mean-pooled transformer) + type/numeric fusion + GNN.

    In dev mode the span encoder is a small BERT; the full-scale config uses
    BERT base.
    The output dimension per node is gnn.hidden.
    """

    def __init__(self, span_encoder_name: str = "bert-base-uncased",
                 hidden: int = 768, num_layers: int = 3, num_relations: int = len(EDGE_TYPES)):
        super().__init__()
        from transformers import AutoModel
        self.span_encoder = AutoModel.from_pretrained(span_encoder_name)
        in_dim = self.span_encoder.config.hidden_size
        self.gnn = RelationGNN(in_dim, hidden=hidden, num_layers=num_layers,
                               num_relations=num_relations)

    @torch.no_grad()
    def span_embeddings(self, tokenizer, texts: List[str], device="cpu") -> torch.Tensor:
        """Mean-pooled wordpiece embeddings per node span -> (N, in_dim)."""
        enc = tokenizer(texts, padding=True, truncation=True, max_length=64,
                        return_tensors="pt").to(device)
        out = self.span_encoder(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        return (out * mask).sum(1) / mask.sum(1).clamp(min=1e-6)

    @torch.no_grad()
    def encode_graph(self, tokenizer, sdg: SDG, device="cpu") -> Dict[str, torch.Tensor]:
        """Full pipeline: span emb ⊕ type ⊕ num -> relation-aware GNN -> {node_id: h}.

        No-grad: the GNN is trained separately (scripts/train_relf.py); its
        embeddings are frozen features for the policy prefix and rewards.
        """
        if not sdg.nodes:
            return {}
        texts = [n.text for n in sdg.nodes]
        span = self.span_embeddings(tokenizer, texts, device=device).to(device)
        nums = torch.stack([numeric_features(n.num_value) for n in sdg.nodes]).to(device)
        tids = torch.tensor([node_type_id(n.type) for n in sdg.nodes], device=device)
        node_init = torch.cat([span, nums], dim=-1)
        id2i = {n.id: i for i, n in enumerate(sdg.nodes)}
        if sdg.edges:
            ei = torch.tensor([[id2i[e.src] for e in sdg.edges],
                               [id2i[e.dst] for e in sdg.edges]], dtype=torch.long, device=device)
            et = torch.tensor([EDGE_TYPES.index(e.type) for e in sdg.edges], device=device)
        else:
            ei = torch.zeros(2, 0, dtype=torch.long, device=device)
            et = torch.zeros(0, dtype=torch.long, device=device)
        h = self.gnn(node_init, tids, ei, et)
        return {n.id: h[i] for i, n in enumerate(sdg.nodes)}


def zero_node_init(node_ids: List[str], hidden: int) -> Dict[str, torch.Tensor]:
    return {nid: torch.zeros(hidden) for nid in node_ids}
