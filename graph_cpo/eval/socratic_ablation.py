"""Controlled ablations of Socratic sub-questions.

Granularity:  {q}^k uses a random fraction k of the sub-questions.
Order:        shuffle({q}) permutes question order.
Relevance:    <base-ques> uses untuned-model questions.
Graph structure: No Graph / Keyword-based / Pruned Graph / Full Graph, each
with its path density (fraction of gold dependencies represented), NLI-judged
error rate, and downstream accuracy.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from ..sdg.schema import SDG, SDGEdge
from ..trajectory import Trajectory
from .protocol_a import random_questions
from .solver import SolverHarness, answer_correct, extract_answer, _gold_from


def fraction(questions: List[str], k: float, seed: int = 0) -> List[str]:
    if k <= 0:
        return []
    rng = random.Random(seed)
    n = max(1, int(round(k * len(questions))))
    return rng.sample(questions, min(n, len(questions)))


def shuffled(questions: List[str], seed: int = 0) -> List[str]:
    out = list(questions)
    random.Random(seed).shuffle(out)
    return out


@dataclass
class AblationRow:
    variation: str
    accuracy: float
    delta: Optional[float] = None


def granularity_ablation(problems: List[Dict], solver: SolverHarness,
                         generator: Callable[[SDG], Trajectory],
                         ks: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)) -> List[AblationRow]:
    rows = []
    full_acc = _augmented_accuracy(problems, solver, generator)
    for k in ks:
        acc = _augmented_accuracy(
            problems, solver, generator,
            transform=lambda qs, pid: fraction(qs, k, seed=hash(pid) % 9999))
        rows.append(AblationRow(variation=f"MWP w/ q^{k:g}", accuracy=acc,
                                delta=acc - full_acc if k < 1.0 else None))
    return rows


def order_ablation(problems: List[Dict], solver: SolverHarness,
                   generator: Callable[[SDG], Trajectory]) -> AblationRow:
    full = _augmented_accuracy(problems, solver, generator)
    acc = _augmented_accuracy(
        problems, solver, generator,
        transform=lambda qs, pid: shuffled(qs, seed=hash(pid) % 9999))
    return AblationRow(variation="MWP w/ shuffle({q})", accuracy=acc, delta=acc - full)


def relevance_ablation(problems: List[Dict], solver: SolverHarness) -> AblationRow:
    """Generic (untuned) questions — often worse than no scaffolding at all."""
    correct = 0
    for item in problems:
        pid = item.get("id", item["question"][:40])
        qs = random_questions(item["question"], seed=hash(pid) % 9999)
        out = solver.solve(item["question"], qs)
        correct += int(answer_correct(extract_answer(out),
                                      _gold_from(item.get("gold_answer") or item.get("answer"))))
    return AblationRow(variation="MWP w/ <base-ques>",
                       accuracy=correct / max(len(problems), 1))


# --------------------------------------------------------------- graph variants

def no_graph_controls(sdg: SDG) -> SDG:
    """No Graph: strip every edge (planner has no dependency structure)."""
    stripped = SDG(problem_id=sdg.problem_id, text=sdg.text,
                   nodes=list(sdg.nodes), edges=[], gold_answer=sdg.gold_answer)
    return stripped


def keyword_graph(sdg: SDG) -> SDG:
    """Keyword-based: connect quantities to the goal through cue words only
    (~40% path density in experiments)."""
    edges = [SDGEdge(src=n.id, dst=sdg.goal_nodes()[0].id, type="goal_dep", conf=0.5)
             for n in sdg.nodes if n.type == "quantity"] if sdg.goal_nodes() else []
    return SDG(problem_id=sdg.problem_id, text=sdg.text, nodes=list(sdg.nodes),
               edges=edges, gold_answer=sdg.gold_answer)


def pruned_graph(sdg: SDG, keep_ratio: float = 0.65) -> SDG:
    """Pruned Graph: keep only the top-confidence edges (~65% path density)."""
    edges = sorted(sdg.edges, key=lambda e: e.conf, reverse=True)
    keep = edges[: max(1, int(len(edges) * keep_ratio))]
    return SDG(problem_id=sdg.problem_id, text=sdg.text, nodes=list(sdg.nodes),
               edges=keep, gold_answer=sdg.gold_answer)


GRAPH_VARIANTS = {
    "no_graph": no_graph_controls,
    "keyword": keyword_graph,
    "pruned": pruned_graph,
    "full": lambda g: g,
}


def graph_structure_ablation(problems: List[Dict], solver: SolverHarness,
                             generator: Callable[[SDG], Trajectory],
                             nli_error_judge: Optional[Callable] = None
                             ) -> List[Dict]:
    """Accuracy + NLI-judged dependency-order error rate per graph strategy."""
    rows = []
    for name, variant in GRAPH_VARIANTS.items():
        correct = errors = total = 0
        for item in problems:
            sdg = item.get("sdg")
            if sdg is None:
                continue
            traj = generator(variant(sdg))
            qs = traj.questions
            out = solver.solve(item["question"], qs)
            correct += int(answer_correct(extract_answer(out),
                                          _gold_from(item.get("gold_answer") or item.get("answer"))))
            if nli_error_judge is not None:
                errors += int(not nli_error_judge(item["question"], qs, sdg))
            total += 1
        rows.append({
            "strategy": name,
            "accuracy": correct / max(total, 1),
            "error_rate": errors / max(total, 1) if nli_error_judge is not None else None,
            "num": total,
        })
    return rows


def _augmented_accuracy(problems: List[Dict], solver: SolverHarness,
                        generator: Callable[[SDG], Trajectory],
                        transform=None) -> float:
    correct = 0
    for item in problems:
        sdg = item.get("sdg")
        pid = item.get("id", item["question"][:40])
        qs = generator(sdg).questions if sdg is not None else []
        if transform is not None:
            qs = transform(qs, pid)
        out = solver.solve(item["question"], qs)
        correct += int(answer_correct(extract_answer(out),
                                      _gold_from(item.get("gold_answer") or item.get("answer"))))
    return correct / max(len(problems), 1)
