"""Silver SDG labels distilled from GSM8K/MathQA solution steps.

GSM8K solutions embed intermediate calculations as ``<<expr=result>>`` markers,
which give us Operation nodes with typed operands and derived Variable nodes for
free. Quantity/Entity/Condition/Goal nodes are extracted from the question text
with rule-based patterns (numbers, arithmetic cue words). These labels are
*silver*: noisy but plentiful (target: 12,000 problems), and used only to train
the tagger and relation classifier — never for the 1,000 expert eval graphs.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .schema import SDG, SDGEdge, SDGNode

# Arithmetic cue words that indicate Condition/Operation nodes.
CUE_WORDS = [
    "twice", "thrice", "each", "left", "more than", "less than", "fewer than",
    "total", "remainder", "average", "ratio", "per", "half", "double", "after",
    "before", "every", "percent", "round up", "round down",
]

_NUMBER_RE = re.compile(r"(?<![\w.])(\d[\d,]*\.?\d*)(?![\w])")
_EXPR_RE = re.compile(r"<<([^=<>]+)=([-\d\.,]+)>>")
_ANSWER_RE = re.compile(r"####\s*([-\d\.,]+)")


def _norm_num(s: str) -> float:
    return float(s.replace(",", ""))


def extract_silver_sdg(problem_id: str, question: str, answer: str) -> SDG:
    """Build a silver SDG from one GSM8K-style (question, solution) pair."""
    sdg = SDG(problem_id=problem_id, text=question, gold_answer=_gold(answer))
    counter = {"n": 0}

    def nid() -> str:
        counter["n"] += 1
        return f"v{counter['n']}"

    # --- Goal node: the interrogative sentence of the question.
    goal = SDGNode(id=nid(), type="goal", text=question.strip())
    sdg.nodes.append(goal)

    # --- Quantity nodes: distinct numbers in the question, with owner context.
    seen = set()
    for m in _NUMBER_RE.finditer(question):
        val = _norm_num(m.group(1))
        if val in seen:
            continue
        seen.add(val)
        lo = max(0, m.start() - 30)
        hi = min(len(question), m.end() + 15)
        sdg.nodes.append(SDGNode(
            id=nid(), type="quantity", text=question[lo:hi].strip(),
            start=m.start(), end=m.end(), num_value=val,
        ))

    # --- Condition nodes: cue-word spans in the question.
    qlower = question.lower()
    for cue in CUE_WORDS:
        idx = qlower.find(cue)
        if idx >= 0:
            sdg.nodes.append(SDGNode(
                id=nid(), type="condition", text=question[idx:idx + len(cue)],
                start=idx, end=idx + len(cue),
            ))

    # --- Operation + derived Variable nodes from <<expr=result>> steps.
    prev_op_id = None
    for expr, res in _EXPR_RE.findall(answer):
        op_type = _classify_op(expr)
        op = SDGNode(id=nid(), type="operation", text=f"Op({op_type}): {expr.strip()}")
        sdg.nodes.append(op)
        var = SDGNode(id=nid(), type="variable", text=f"result of {expr.strip()} = {res}",
                      num_value=_norm_num(res))
        sdg.nodes.append(var)
        sdg.edges.append(SDGEdge(src=op.id, dst=var.id, type="derive", conf=0.95))
        # link op inputs: question quantities mentioned inside the expression
        for qn in [n for n in sdg.nodes if n.type == "quantity"]:
            tokens = [t.strip() for t in re.split(r"[+\-*/]", expr)]
            if qn.num_value is not None and any(
                    abs(qn.num_value - _safe_float(t)) < 1e-9 for t in tokens if _safe_float(t) is not None):
                sdg.edges.append(SDGEdge(src=qn.id, dst=op.id, type="op_input", conf=0.9))
        # chain: sequential solution steps depend on previous derived values
        if prev_op_id is not None:
            sdg.edges.append(SDGEdge(src=prev_op_id, dst=op.id, type="prerequisite", conf=0.85))
        prev_op_id = op.id

    # --- Connect the last derived variable to the goal.
    derived = [n for n in sdg.nodes if n.type == "variable"]
    if derived:
        sdg.edges.append(SDGEdge(src=derived[-1].id, dst=goal.id, type="goal_dep", conf=0.95))

    # --- Conditions constrain the operations/goal.
    conditions = [n for n in sdg.nodes if n.type == "condition"]
    operations = [n for n in sdg.nodes if n.type == "operation"]
    for c in conditions:
        target = operations[-1] if operations else goal
        sdg.edges.append(SDGEdge(src=c.id, dst=target.id, type="condition_edge", conf=0.7))
        # cue-driven operation creation when no <<expr>> steps exist at all
    if not operations and len(sdg.nodes) > 1:
        # fallback: an implicit operation between quantities and the goal
        op = SDGNode(id=nid(), type="operation", text="Op(implicit)")
        sdg.nodes.append(op)
        for qn in [n for n in sdg.nodes if n.type == "quantity"]:
            sdg.edges.append(SDGEdge(src=qn.id, dst=op.id, type="op_input", conf=0.6))
        sdg.edges.append(SDGEdge(src=op.id, dst=goal.id, type="goal_dep", conf=0.6))

    # --- quantity-of edges: attribute each quantity to the text's main entity noun.
    entity_word = _guess_entity(question)
    if entity_word:
        ent = SDGNode(id=nid(), type="entity", text=entity_word)
        sdg.nodes.append(ent)
        for qn in [n for n in sdg.nodes if n.type == "quantity"]:
            sdg.edges.append(SDGEdge(src=qn.id, dst=ent.id, type="quantity_of", conf=0.8))

    from .dag import to_dag
    return to_dag(sdg)


def _gold(answer: str) -> Optional[str]:
    m = _ANSWER_RE.search(answer)
    return m.group(1) if m else None


def _safe_float(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def _classify_op(expr: str) -> str:
    expr = expr.strip()
    if "+" in expr:
        return "+"
    if re.search(r"(?<!\d)-", expr):
        return "-"
    if "*" in expr or re.search(r"\d\s*x\s*\d", expr):
        return "*"
    if "/" in expr:
        return "/"
    return "?"


_ENTITY_STOP = {
    "how", "many", "much", "what", "the", "a", "an", "is", "are", "was", "were",
    "will", "would", "if", "then", "and", "or", "to", "of", "in", "for", "each",
    "there", "this", "that", "her", "his", "their", "he", "she", "they", "it",
}


def _guess_entity(question: str) -> Optional[str]:
    """Pick the most frequent plural/singular noun-ish token after a determiner."""
    words = re.findall(r"[A-Za-z]+", question)
    candidates = [w.lower() for w in words if w.lower() not in _ENTITY_STOP and len(w) > 2]
    if not candidates:
        return None
    from collections import Counter
    return Counter(candidates).most_common(1)[0][0]
