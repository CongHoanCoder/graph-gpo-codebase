"""Silver label extraction tests (rule-based, no model downloads)."""
import unittest

from graph_cpo.sdg.dag import is_dag
from graph_cpo.sdg.silver_labels import extract_silver_sdg

CUPCAKES = ("Sarah has 24 cupcakes. She gives 8 cupcakes to her friend Tom and then "
            "bakes 15 more cupcakes. If she wants to pack all her cupcakes into boxes "
            "that hold 5 cupcakes each, how many boxes will she need?")
SOLUTION = ("She gives away 8, so 24-8=<<24-8=16>>16 cupcakes remain.\n"
            "Then she bakes 15 more: 16+15=<<16+15=31>>31 cupcakes.\n"
            "Boxes needed: 31/5=<<31/5=6.2>>6.2, round up. #### 7")


class TestSilverLabels(unittest.TestCase):
    def setUp(self):
        self.sdg = extract_silver_sdg("cupcakes", CUPCAKES, SOLUTION)

    def test_produces_a_dag(self):
        self.assertTrue(is_dag(self.sdg))

    def test_goal_node_exists(self):
        self.assertEqual(len(self.sdg.goal_nodes()), 1)

    def test_quantities_extracted(self):
        vals = {n.num_value for n in self.sdg.nodes if n.type == "quantity"}
        self.assertIn(24.0, vals)
        self.assertIn(8.0, vals)
        self.assertIn(15.0, vals)
        self.assertIn(5.0, vals)

    def test_operation_nodes_from_solution_steps(self):
        ops = [n for n in self.sdg.nodes if n.type == "operation"]
        self.assertGreaterEqual(len(ops), 2)   # 24-8 and 16+15 at least

    def test_derived_variables_linked_to_goal(self):
        goal = self.sdg.goal_nodes()[0]
        dep_types = {e.type for e in self.sdg.incoming(goal.id)}
        self.assertIn("goal_dep", dep_types)

    def test_gold_answer_parsed(self):
        self.assertEqual(self.sdg.gold_answer, "7")


if __name__ == "__main__":
    unittest.main()
