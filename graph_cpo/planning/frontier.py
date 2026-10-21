"""Frontier tracking and node resolution.

F_t = { v in V_unres ∩ V_Q : Pre(v) ⊆ V_res }

A node is marked resolved only when the generated question passes BOTH gates:
  cos(Enc(q), h_v)      >= tau_sim   (semantic alignment)
  P_NLI(entail | ...)   >= tau_nli   (dependency addressed)
Otherwise the node stays unresolved and the question gets a lower reward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

from ..sdg.schema import SDG, V_Q_TYPES


@dataclass
class GraphState:
    """G_t = (G, V_res, V_unres)."""
    sdg: SDG
    resolved: Set[str] = field(default_factory=set)

    @property
    def unresolved(self) -> Set[str]:
        return set(self.sdg.node_ids()) - self.resolved

    def frontier(self) -> List[str]:
        """Targetable reasoning frontier F_t."""
        out = []
        for n in self.sdg.targetable():
            if n.id in self.resolved:
                continue
            pre = set(self.sdg.prerequisites(n.id))
            if pre.issubset(self.resolved):
                out.append(n.id)
        return out


class FrontierTracker:
    """Advances the graph state as questions are generated.

    resolve_gate(q_text, node_id) -> bool is injected so this module stays free
    of model dependencies (the gate is cos-sim + NLI, built in rewards/).
    """

    def __init__(self, sdg: SDG, resolve_gate: Optional[Callable[[str, str], bool]] = None):
        self.state = GraphState(sdg=sdg)
        self.resolve_gate = resolve_gate or _always_false

    def frontier(self) -> List[str]:
        return self.state.frontier()

    def step(self, question: str, target_node: str) -> bool:
        """Apply resolution gate; return True if the node was marked resolved."""
        if self.resolve_gate(question, target_node):
            self.state.resolved.add(target_node)
            return True
        return False

    def force_resolve(self, node_id: str) -> None:
        """Teacher forcing: gold trajectories resolve their target unconditionally."""
        self.state.resolved.add(node_id)


def _always_false(question: str, node_id: str) -> bool:
    return False


def resolved_desc(tracker: FrontierTracker) -> str:
    """Human-readable resolved-node summary for the Avoid field of action frames."""
    sdg = tracker.state.sdg
    parts = []
    for nid in sorted(tracker.state.resolved):
        try:
            parts.append(sdg.node(nid).text)
        except KeyError:
            pass
    return "; ".join(parts)
