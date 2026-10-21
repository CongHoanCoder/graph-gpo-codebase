"""SDG construction quality vs. expert-annotated graphs.

Metrics:
  - Entity boundary F1 / type accuracy (span-overlap matching)
  - per-type F1 (Quantity / Operation / Variable)
  - Edge precision / recall / F1 and relation type accuracy
  - Critical Path Preservation: fraction of gold solution-critical (longest
    prerequisite path to the goal) dependencies recovered in the prediction
  - Normalized Graph Edit Distance
  - Topological Correctness: fraction of gold prerequisite-order pairs that the
    predicted graph respects
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from ..sdg.dag import to_dag
from ..sdg.schema import SDG


def _spans_overlap(a_start, a_end, b_start, b_end, iou_thresh: float = 0.5) -> bool:
    if a_start < 0 or b_start < 0:
        return False
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = (a_end - a_start) + (b_end - b_start) - inter
    return union > 0 and inter / union >= iou_thresh


def _match_nodes(pred: SDG, gold: SDG, iou_thresh: float = 0.5
                 ) -> Tuple[List[Tuple[str, str]], List[str], List[str]]:
    """Match predicted to gold nodes by span overlap (fallback: exact text)."""
    pairs: List[Tuple[str, str]] = []
    used_gold: Set[str] = set()
    for pn in pred.nodes:
        best, best_score = None, 0.0
        for gn in gold.nodes:
            if gn.id in used_gold:
                continue
            if _spans_overlap(pn.start, pn.end, gn.start, gn.end, iou_thresh):
                score = 1.0
            elif pn.text.strip().lower() == gn.text.strip().lower() and pn.text:
                score = 0.8
            else:
                continue
            if score > best_score:
                best, best_score = gn.id, score
        if best is not None:
            pairs.append((pn.id, best))
            used_gold.add(best)
    matched_pred = {p for p, _ in pairs}
    matched_gold = {g for _, g in pairs}
    miss_pred = [n.id for n in pred.nodes if n.id not in matched_pred]
    miss_gold = [n.id for n in gold.nodes if n.id not in matched_gold]
    return pairs, miss_pred, miss_gold


@dataclass
class GraphQuality:
    boundary_precision: float
    boundary_recall: float
    boundary_f1: float
    type_accuracy: float
    per_type_f1: Dict[str, float]
    edge_precision: float
    edge_recall: float
    edge_f1: float
    relation_type_accuracy: float
    critical_path_preservation: float
    normalized_graph_edit_distance: float
    topological_correctness: float


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def critical_path(gold: SDG) -> List[Tuple[str, str]]:
    """Longest prerequisite chain ending at a goal node (solution-critical path)."""
    dag_gold = to_dag(SDG(problem_id=gold.problem_id, text=gold.text,
                          nodes=list(gold.nodes), edges=list(gold.edges)))
    g = nx.DiGraph()
    g.add_nodes_from(dag_gold.node_ids())
    g.add_edges_from((e.src, e.dst) for e in dag_gold.edges)
    goals = [n.id for n in gold.goal_nodes()]
    if not goals:
        return []
    # lengths[x] = distance from x to the goal in the original graph
    best_node, best_depth = None, -1
    for goal in goals:
        lengths = nx.single_source_shortest_path_length(g.reverse(), goal)
        depth = max(lengths.values(), default=0)
        if depth > best_depth:
            best_node, best_depth = goal, depth
    lengths = nx.single_source_shortest_path_length(g.reverse(), best_node)
    # walk backwards from the goal along the longest chain (increasing distance)
    path = [best_node]
    cur = best_node
    while True:
        preds = [p for p in g.predecessors(cur)]
        if not preds:
            break
        nxt = max(preds, key=lambda p: lengths.get(p, -1))
        if lengths.get(nxt, -1) <= lengths.get(cur, 0):
            break
        path.append(nxt)
        cur = nxt
    path.reverse()
    return list(zip(path[:-1], path[1:]))


def graph_quality(pred: SDG, gold: SDG, iou_thresh: float = 0.5) -> GraphQuality:
    pairs, miss_pred, miss_gold = _match_nodes(pred, gold, iou_thresh)
    n_pred, n_gold = len(pred.nodes), len(gold.nodes)
    tp = len(pairs)
    precision = tp / max(n_pred, 1)
    recall = tp / max(n_gold, 1)

    p2g = dict(pairs)
    g2p = {g: p for p, g in pairs}
    type_correct = sum(
        1 for p, g in pairs if pred.node(p).type == gold.node(g).type)
    type_acc = type_correct / max(tp, 1)

    per_type: Dict[str, float] = {}
    for t in {n.type for n in gold.nodes} | {n.type for n in pred.nodes}:
        g_t = {n.id for n in gold.nodes if n.type == t}
        p_t = {n.id for n in pred.nodes if n.type == t}
        hit = len({p2g[p] for p, _ in pairs if pred.node(p).type == t
                   and p2g[p] in g_t})
        p_ = hit / max(len(p_t), 1)
        r_ = hit / max(len(g_t), 1)
        per_type[t] = _f1(p_, r_)

    gold_edges = {(e.src, e.dst, e.type) for e in gold.edges}
    pred_edges_mapped = {(p2g.get(e.src, f"?{e.src}"), p2g.get(e.dst, f"?{e.dst}"),
                         e.type) for e in pred.edges
                         if e.src in p2g and e.dst in p2g}
    edge_tp = len(pred_edges_mapped & gold_edges)
    e_p = edge_tp / max(len(pred_edges_mapped), 1)
    e_r = edge_tp / max(len(gold_edges), 1)
    type_correct_edges = len({e for e in pred_edges_mapped if e in gold_edges})
    rel_acc = type_correct_edges / max(len(pred_edges_mapped), 1)

    # critical path preservation
    cpath = critical_path(gold)
    preserved = sum(1 for (u, v) in cpath
                    if (g2p.get(u), g2p.get(v)) in
                    {(e.src, e.dst) for e in pred.edges} or
                    any(e.src == g2p.get(u) and e.dst == g2p.get(v)
                        for e in pred.edges))
    cpp = preserved / max(len(cpath), 1)

    # normalized graph edit distance (node + edge symmetric differences)
    sym_nodes = len(miss_pred) + len(miss_gold)
    all_gold_e = {(u, v) for u, v, _ in gold_edges}
    all_pred_e = {(e.src, e.dst) for e in pred.edges}
    mapped_pred_e = {(p2g.get(u), p2g.get(v)) for u, v in all_pred_e
                     if u in p2g and v in p2g}
    sym_edges = len(all_gold_e - mapped_pred_e) + len(mapped_pred_e - all_gold_e)
    denom = (n_gold + len(all_gold_e)) or 1
    ngd = (sym_nodes + sym_edges) / denom

    # topological correctness: gold prerequisite pairs respected by prediction
    gold_order = set()
    for nid in gold.node_ids():
        for pre in gold.prerequisites(nid):
            gold_order.add((pre, nid))
    pg = nx.DiGraph()
    pg.add_nodes_from(g2p.keys())
    for e in pred.edges:
        if e.src in p2g and e.dst in p2g:
            pg.add_edge(p2g[e.src], p2g[e.dst])
    try:
        reach = dict(nx.all_pairs_shortest_path_length(pg))
    except Exception:
        reach = {}
    ok = sum(1 for (u, v) in gold_order
             if v in reach.get(u, {}) or not pg.has_edge(u, v))
    topo = ok / max(len(gold_order), 1)

    return GraphQuality(
        boundary_precision=precision, boundary_recall=recall,
        boundary_f1=_f1(precision, recall), type_accuracy=type_acc,
        per_type_f1=per_type, edge_precision=e_p, edge_recall=e_r,
        edge_f1=_f1(e_p, e_r), relation_type_accuracy=rel_acc,
        critical_path_preservation=cpp,
        normalized_graph_edit_distance=ngd, topological_correctness=topo)


def evaluate_against_expert(preds: List[SDG], golds: List[SDG]) -> Dict[str, float]:
    """Corpus-level averages over expert-annotated pairs (matched by problem_id)."""
    gold_by_id = {g.problem_id: g for g in golds}
    rows = [graph_quality(p, gold_by_id[p.problem_id])
            for p in preds if p.problem_id in gold_by_id]
    if not rows:
        return {}
    import numpy as np
    return {
        "boundary_f1": float(np.mean([r.boundary_f1 for r in rows])),
        "type_accuracy": float(np.mean([r.type_accuracy for r in rows])),
        "edge_f1": float(np.mean([r.edge_f1 for r in rows])),
        "relation_type_accuracy": float(np.mean([r.relation_type_accuracy for r in rows])),
        "critical_path_preservation": float(np.mean([r.critical_path_preservation for r in rows])),
        "normalized_graph_edit_distance": float(np.mean([r.normalized_graph_edit_distance for r in rows])),
        "topological_correctness": float(np.mean([r.topological_correctness for r in rows])),
        "num_graphs": len(rows),
    }
