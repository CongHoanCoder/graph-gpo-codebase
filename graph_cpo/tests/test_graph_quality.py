"""Graph-quality metric tests (perfect prediction must score ~1.0 / GED ~0)."""
import unittest

from graph_cpo.eval.graph_quality import critical_path, graph_quality
from .test_sdg import toy_sdg


class TestGraphQuality(unittest.TestCase):
    def test_perfect_prediction(self):
        gold = toy_sdg()
        pred = toy_sdg()
        q = graph_quality(pred, gold)
        self.assertAlmostEqual(q.boundary_f1, 1.0)
        self.assertAlmostEqual(q.type_accuracy, 1.0)
        self.assertAlmostEqual(q.edge_f1, 1.0)
        self.assertAlmostEqual(q.critical_path_preservation, 1.0)
        self.assertAlmostEqual(q.normalized_graph_edit_distance, 0.0)
        self.assertAlmostEqual(q.topological_correctness, 1.0)

    def test_missing_nodes_penalized(self):
        gold = toy_sdg()
        pred = toy_sdg()
        pred.nodes = pred.nodes[:-2]      # drop op2 + goal
        pred.edges = [e for e in pred.edges
                      if e.src in {n.id for n in pred.nodes}
                      and e.dst in {n.id for n in pred.nodes}]
        q = graph_quality(pred, gold)
        self.assertLess(q.boundary_f1, 1.0)
        self.assertLess(q.edge_recall, 1.0)
        self.assertGreater(q.normalized_graph_edit_distance, 0.0)

    def test_critical_path(self):
        g = toy_sdg()
        path = critical_path(g)
        # longest chain q12 -> op1 -> sub -> op2 -> goal
        self.assertEqual(len(path), 4)
        self.assertEqual(path[-1], ("op2", "goal"))


if __name__ == "__main__":
    unittest.main()
