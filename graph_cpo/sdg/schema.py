"""Schema of the Semantic Dependency Graph.

Node types:  Quantity, Entity, Variable, Operation, Condition, Goal
Edge types:  quantity-of, op-input, derive, condition, prerequisite, goal-dep
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

NODE_TYPES = ["quantity", "entity", "variable", "operation", "condition", "goal"]
EDGE_TYPES = ["quantity_of", "op_input", "derive", "condition_edge", "prerequisite", "goal_dep"]

# V_Q: targetable node types (the frontier excludes purely entity nodes).
V_Q_TYPES = {"quantity", "variable", "operation", "condition", "goal"}

# Pedagogical intents derived from node type (used in action frames).
INTENT_BY_TYPE = {
    "quantity": "identify-quantity",
    "entity": "identify-entity",
    "variable": "derive-value",
    "operation": "select-operation",
    "condition": "interpret-condition",
    "goal": "derive-value",
}


def node_type_id(t: str) -> int:
    return NODE_TYPES.index(t)


def edge_type_id(t: str) -> int:
    return EDGE_TYPES.index(t)


@dataclass
class SDGNode:
    id: str
    type: str                       # one of NODE_TYPES
    text: str                       # surface span
    start: int = -1                 # char offset in problem text (-1 if not a span)
    end: int = -1
    num_value: Optional[float] = None   # normalized numeric value (Quantity nodes)
    unit: str = ""                  # owner/unit of a quantity

    def __post_init__(self):
        assert self.type in NODE_TYPES, f"bad node type {self.type}"


@dataclass
class SDGEdge:
    src: str
    dst: str
    type: str                       # one of EDGE_TYPES
    conf: float = 1.0               # relation confidence w_e in [0, 1]

    def __post_init__(self):
        assert self.type in EDGE_TYPES, f"bad edge type {self.type}"


@dataclass
class SDG:
    """A Semantic Dependency Graph G = (V, E, tau, rho, w)."""
    problem_id: str
    text: str                       # MWP text P
    nodes: List[SDGNode] = field(default_factory=list)
    edges: List[SDGEdge] = field(default_factory=list)
    # optional gold answer ("#### 42" style) for supervision / silver labels
    gold_answer: Optional[str] = None

    def node(self, nid: str) -> SDGNode:
        for n in self.nodes:
            if n.id == nid:
                return n
        raise KeyError(nid)

    def node_ids(self) -> List[str]:
        return [n.id for n in self.nodes]

    def has_node(self, nid: str) -> bool:
        return any(n.id == nid for n in self.nodes)

    def outgoing(self, nid: str, etype: Optional[str] = None) -> List[SDGEdge]:
        return [e for e in self.edges if e.src == nid and (etype is None or e.type == etype)]

    def incoming(self, nid: str, etype: Optional[str] = None) -> List[SDGEdge]:
        return [e for e in self.edges if e.dst == nid and (etype is None or e.type == etype)]

    def prerequisites(self, nid: str) -> List[str]:
        """Pre(v): nodes that must be resolved before v (prerequisite, op-input,
        condition and goal-dependency edges all encode pedagogical ordering)."""
        deps = set()
        for e in self.incoming(nid):
            if e.type in ("prerequisite", "op_input", "condition_edge",
                          "goal_dep", "derive"):
                deps.add(e.src)
        return sorted(deps)

    def children(self, nid: str) -> List[str]:
        kids = set()
        for e in self.outgoing(nid):
            if e.type in ("prerequisite", "op_input", "derive", "goal_dep"):
                kids.add(e.dst)
        return sorted(kids)

    def goal_nodes(self) -> List[SDGNode]:
        return [n for n in self.nodes if n.type == "goal"]

    def targetable(self) -> List[SDGNode]:
        return [n for n in self.nodes if n.type in V_Q_TYPES]

    def one_hop(self, nid: str) -> List[str]:
        nbrs = {e.src for e in self.incoming(nid)} | {e.dst for e in self.outgoing(nid)}
        nbrs.discard(nid)
        return sorted(nbrs)

    def incident_confidence(self, nid: str) -> float:
        """Average confidence of edges incident to v (bar{w}(v))."""
        inc = [e.conf for e in self.incoming(nid)] + [e.conf for e in self.outgoing(nid)]
        return sum(inc) / len(inc) if inc else 0.0

    def to_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "text": self.text,
            "gold_answer": self.gold_answer,
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
        }

    @staticmethod
    def from_dict(d: dict) -> "SDG":
        return SDG(
            problem_id=d["problem_id"],
            text=d["text"],
            gold_answer=d.get("gold_answer"),
            nodes=[SDGNode(**n) for n in d["nodes"]],
            edges=[SDGEdge(**e) for e in d["edges"]],
        )
