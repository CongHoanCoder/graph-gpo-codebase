"""DAG utilities: cycle detection/removal, topological depth, goal distances.

Cycle handling: if the predicted graph contains a cycle, remove the
lowest-confidence edge on a cycle until a valid Directed Acyclic Graph is
obtained.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx

from .schema import SDG


def _nx_graph(sdg: SDG) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(sdg.node_ids())
    for e in sdg.edges:
        # parallel edges collapse; keep the highest-confidence one
        if g.has_edge(e.src, e.dst):
            if e.conf > g[e.src][e.dst]["conf"]:
                g[e.src][e.dst].update(type=e.type, conf=e.conf)
        else:
            g.add_edge(e.src, e.dst, type=e.type, conf=e.conf)
    return g


def is_dag(sdg: SDG) -> bool:
    return nx.is_directed_acyclic_graph(_nx_graph(sdg))


def lowest_confidence_cycle_edge(sdg: SDG) -> Optional[Tuple[str, str]]:
    """Return (src, dst) of the lowest-confidence edge on any cycle, or None."""
    g = _nx_graph(sdg)
    try:
        cycle = nx.find_cycle(g)
    except nx.NetworkXNoCycle:
        return None
    # networkx returns 2- or 3-tuples depending on version/graph
    edges = [(e[0], e[1]) for e in cycle]
    return min(edges, key=lambda uv: g[uv[0]][uv[1]]["conf"])


def to_dag(sdg: SDG) -> SDG:
    """Remove lowest-confidence cycle edges until the graph is acyclic (in place)."""
    while not is_dag(sdg):
        edge = lowest_confidence_cycle_edge(sdg)
        if edge is None:
            break
        src, dst = edge
        sdg.edges = [e for e in sdg.edges if not (e.src == src and e.dst == dst)]
    return sdg


def topological_depth(sdg: SDG) -> Dict[str, int]:
    """depth(v): longest prerequisite-chain length from a source node to v."""
    g = _nx_graph(sdg)
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("topological_depth requires a DAG; call to_dag() first")
    depth: Dict[str, int] = {n: 0 for n in g.nodes}
    for u in nx.topological_sort(g):
        for v in g.successors(u):
            depth[v] = max(depth[v], depth[u] + 1)
    return depth


def shortest_goal_distance(sdg: SDG) -> Dict[str, float]:
    """dist(v, g): shortest directed distance from v to the nearest goal node.

    Nodes that cannot reach any goal get distance = +inf (score term 1/(1+inf)=0).
    """
    g = _nx_graph(sdg)
    goals = [n.id for n in sdg.goal_nodes()]
    dist: Dict[str, float] = {n: float("inf") for n in g.nodes}
    if not goals:
        return dist
    # run BFS backwards from goal nodes on the reversed graph
    reverse = g.reverse()
    import collections
    for goal in goals:
        dist[goal] = min(dist[goal], 0.0)
        queue = collections.deque([goal])
        seen = {goal}
        while queue:
            u = queue.popleft()
            for v in reverse.successors(u):
                d = dist[u] + 1
                if d < dist[v]:
                    dist[v] = d
                if v not in seen:
                    seen.add(v)
                    queue.append(v)
    return dist
