"""Graph prefix projection:

p_t = W_p [ h_{v_t} ⊕ Pool_{u in N_1(v_t)} h_u ]

The prefix is a short sequence of continuous embedding vectors prepended to the
policy's token embeddings, giving the generator structural information that the
serialized text frame cannot carry.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from ..sdg.schema import SDG


class GraphPrefixProjector(nn.Module):
    def __init__(self, gnn_hidden: int, prefix_dim: int, num_prefix_tokens: int = 4):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.proj = nn.Linear(2 * gnn_hidden, prefix_dim * num_prefix_tokens)

    def forward(self, target_emb: torch.Tensor,
                neighbor_embs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns (num_prefix_tokens, prefix_dim)."""
        if neighbor_embs is not None and neighbor_embs.numel() > 0:
            pooled = neighbor_embs.mean(dim=0)
        else:
            pooled = torch.zeros_like(target_emb)
        flat = self.proj(torch.cat([target_emb, pooled], dim=-1))
        return flat.view(self.num_prefix_tokens, -1)

    def node_neighbors(self, sdg: SDG, node_emb: Dict[str, torch.Tensor],
                       node_id: str) -> Optional[torch.Tensor]:
        nbrs = sdg.one_hop(node_id)
        embs = [node_emb[n] for n in nbrs if n in node_emb]
        if not embs:
            return None
        return torch.stack(embs)
