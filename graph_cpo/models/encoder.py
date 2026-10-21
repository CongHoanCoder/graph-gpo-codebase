"""Frozen sentence encoder for the contrastive relevance reward.

Enc(·) embeds generated questions and node verbalizations in ONE shared space,
so cos(Enc(q_t), Enc(d(v_t))) is well defined. With rel_space=text (default)
both sides use the same frozen encoder — see rewards/contrastive.py. The option
rel_space=gnn_projected instead projects GNN embeddings into this space.
"""
from __future__ import annotations

from typing import List, Optional, Union

import torch
import torch.nn as nn


class SentenceEncoder:
    """Thin wrapper over a sentence-transformers style model (mean-pooled)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu"):
        from transformers import AutoModel, AutoTokenizer
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, texts: Union[str, List[str]]) -> torch.Tensor:
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        enc = self.tokenizer(batch, padding=True, truncation=True, max_length=128,
                             return_tensors="pt").to(self.device)
        out = self.model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb[0] if single else emb

    @staticmethod
    def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(torch.dot(torch.nn.functional.normalize(a, dim=-1),
                               torch.nn.functional.normalize(b, dim=-1)))


class GNNToEncoderProjector(nn.Module):
    """Learned projection mapping GNN node embeddings into the encoder space
    (used only when rewards.rel_space == 'gnn_projected')."""

    def __init__(self, gnn_hidden: int, enc_dim: int):
        super().__init__()
        self.net = nn.Linear(gnn_hidden, enc_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.net(h), dim=-1)
