"""SDG schema and DAG utility tests."""
import unittest

from graph_cpo.sdg.dag import (is_dag, shortest_goal_distance, to_dag,
                               topological_depth)
from graph_cpo.sdg.schema import SDG, SDGEdge, SDGNode


def toy_sdg() -> SDG:
    """12, 5 -> (+) -> subtotal; subtotal, 4 -> (-) -> goal (illustrative example)."""
    g = SDG(problem_id="t1", text="Maria has 12 apples. She buys 5 more and gives 4 away.")
    g.nodes = [
        SDGNode(id="q12", type="quantity", text="12 apples", num_value=12),
        SDGNode(id="q5", type="quantity", text="5 more", num_value=5),
        SDGNode(id="q4", type="quantity", text="4 away", num_value=4),
        SDGNode(id="op1", type="operation", text="Op(+): 12+5"),
        SDGNode(id="sub", type="variable", text="subtotal", num_value=17),
        SDGNode(id="op2", type="operation", text="Op(-): 17-4"),
        SDGNode(id="goal", type="goal", text="how many apples now?"),
    ]
    g.edges = [
        SDGEdge(src="q12", dst="op1", type="op_input"),
        SDGEdge(src="q5", dst="op1", type="op_input"),
        SDGEdge(src="op1", dst="sub", type="derive"),
        SDGEdge(src="sub", dst="op2", type="op_input"),
        SDGEdge(src="q4", dst="op2", type="op_input"),
        SDGEdge(src="op2", dst="goal", type="goal_dep"),
    ]
    return g


class TestSDGSchema(unittest.TestCase):
    def test_prerequisites(self):
        g = toy_sdg()
        self.assertEqual(g.prerequisites("op2"), ["q4", "sub"])
        self.assertEqual(g.prerequisites("q12"), [])

    def test_targetable_excludes_entities(self):
        g = toy_sdg()
        g.nodes.append(SDGNode(id="e1", type="entity", text="apples"))
        types = {n.type for n in g.targetable()}
        self.assertNotIn("entity", types)

    def test_goal_nodes(self):
        g = toy_sdg()
        self.assertEqual([n.id for n in g.goal_nodes()], ["goal"])


class TestDAG(unittest.TestCase):
    def test_is_dag(self):
        self.assertTrue(is_dag(toy_sdg()))

    def test_cycle_removal(self):
        g = toy_sdg()
        g.edges.append(SDGEdge(src="goal", dst="q12", type="prerequisite", conf=0.2))
        self.assertFalse(is_dag(g))
        to_dag(g)
        self.assertTrue(is_dag(g))
        # the lowest-confidence edge was the one removed
        self.assertFalse(any(e.src == "goal" and e.dst == "q12" for e in g.edges))

    def test_topological_depth(self):
        g = toy_sdg()
        depth = topological_depth(g)
        self.assertEqual(depth["q12"], 0)
        self.assertEqual(depth["op1"], 1)
        self.assertEqual(depth["sub"], 2)
        self.assertEqual(depth["goal"], 4)

    def test_goal_distance(self):
        g = toy_sdg()
        dist = shortest_goal_distance(g)
        self.assertEqual(dist["goal"], 0)
        self.assertEqual(dist["op2"], 1)
        self.assertEqual(dist["q12"], 4)   # q12 -> op1 -> sub -> op2 -> goal


if __name__ == "__main__":
    unittest.main()
