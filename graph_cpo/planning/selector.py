"""Frontier node selection among the targetable nodes of the frontier.

v_t = argmax_v [ lambda_g * 1/(1+dist(v,g))
               + lambda_d * depth(v)
               + lambda_c * bar_w(v)
               - lambda_r * red(v) ]

red(v) = max_{u in V_res} cos(h_v, h_u); zero when V_res is empty.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

import torch

from ..sdg.dag import shortest_goal_distance, topological_depth
from ..sdg.schema import SDG


def node_score(sdg: SDG, node_id: str, goal_dist: float, depth: int,
               redundancy: float, w_avg: float,
               lambda_g: float, lambda_d: float, lambda_c: float, lambda_r: float) -> float:
    dist_term = 1.0 / (1.0 + goal_dist) if math.isfinite(goal_dist) else 0.0
    return (lambda_g * dist_term
            + lambda_d * depth
            + lambda_c * w_avg
            - lambda_r * redundancy)


def select_frontier_node(sdg: SDG, frontier: List[str], resolved: List[str],
                         node_emb: Optional[Dict[str, torch.Tensor]] = None,
                         lambda_g: float = 0.4, lambda_d: float = 0.3,
                         lambda_c: float = 0.2, lambda_r: float = 0.1) -> Optional[str]:
    """Pick v_t among frontier nodes; None if the frontier is empty."""
    if not frontier:
        return None
    gdist = shortest_goal_distance(sdg)
    depth = topological_depth(sdg)

    def red(vid: str) -> float:
        if not resolved or node_emb is None or vid not in node_emb:
            return 0.0
        h = node_emb[vid]
        best = 0.0
        for u in resolved:
            if u in node_emb:
                best = max(best, float(torch.nn.functional.cosine_similarity(
                    h.unsqueeze(0), node_emb[u].unsqueeze(0)).item()))
        return best

    best_id, best_score = None, -float("inf")
    for vid in frontier:
        s = node_score(sdg, vid,
                       goal_dist=gdist.get(vid, float("inf")),
                       depth=depth.get(vid, 0),
                       redundancy=red(vid),
                       w_avg=sdg.incident_confidence(vid),
                       lambda_g=lambda_g, lambda_d=lambda_d,
                       lambda_c=lambda_c, lambda_r=lambda_r)
        if s > best_score:
            best_id, best_score = vid, s
    return best_id
