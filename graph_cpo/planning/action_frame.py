"""Pedagogical action frames: fields and serialized control prompt.

z_t = ( d(v_t), tau(v_t), iota(v_t), d(Pre(v_t)), d(N_1(v_t)), d(V_res) )
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from ..sdg.schema import INTENT_BY_TYPE, SDG


@dataclass
class ActionFrame:
    target_id: str
    target_text: str           # d(v_t)
    node_type: str             # tau(v_t)
    intent: str                # iota(v_t)
    prereq_texts: List[str] = field(default_factory=list)   # d(Pre(v_t))
    neighbor_texts: List[str] = field(default_factory=list)  # d(N_1(v_t))
    resolved_texts: List[str] = field(default_factory=list)  # d(V_res)


def build_action_frame(sdg: SDG, node_id: str, resolved_ids: List[str]) -> ActionFrame:
    node = sdg.node(node_id)
    pre_ids = sdg.prerequisites(node_id)
    nbr_ids = sdg.one_hop(node_id)
    return ActionFrame(
        target_id=node_id,
        target_text=node.text,
        node_type=node.type,
        intent=INTENT_BY_TYPE[node.type],
        prereq_texts=[sdg.node(p).text for p in pre_ids if p in sdg.node_ids()],
        neighbor_texts=[sdg.node(n).text for n in nbr_ids if n in sdg.node_ids()],
        resolved_texts=[sdg.node(r).text for r in resolved_ids if r in sdg.node_ids()],
    )


FRAME_TEMPLATE = (
    "Target: {target}; Intent: {intent}; Prerequisites: {prereqs}; "
    "Local relation: {neighbors}; Avoid: {avoid}. "
    "Ask exactly one Socratic question that helps derive the target "
    "without directly revealing the answer."
)


def serialize_frame(frame: ActionFrame) -> str:
    return FRAME_TEMPLATE.format(
        target=frame.target_text,
        intent=frame.intent,
        prereqs="; ".join(frame.prereq_texts) if frame.prereq_texts else "none",
        neighbors="; ".join(frame.neighbor_texts) if frame.neighbor_texts else "none",
        avoid="; ".join(frame.resolved_texts) if frame.resolved_texts else "nothing",
    )


def verbalize_node(node) -> str:
    """d(·) for a node: its span, normalized value, and unit (used by NLI coherence)."""
    parts = [node.text]
    if node.num_value is not None:
        parts.append(f"(value: {node.num_value:g})")
    if node.unit:
        parts.append(f"(unit/owner: {node.unit})")
    return " ".join(parts)


def verbalize_dependency(frame: ActionFrame) -> str:
    """d(v_t) as a declarative target hypothesis for the NLI coherence judge."""
    return (f"The question should address the {frame.node_type} "
            f"'{frame.target_text}' ({frame.intent})")
