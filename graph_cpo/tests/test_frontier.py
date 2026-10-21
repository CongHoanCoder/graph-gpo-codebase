"""Frontier tracking and node-selection tests."""
import unittest

from graph_cpo.planning.action_frame import build_action_frame, serialize_frame
from graph_cpo.planning.frontier import FrontierTracker
from graph_cpo.planning.selector import select_frontier_node
from .test_sdg import toy_sdg


class TestFrontier(unittest.TestCase):
    def test_initial_frontier_excludes_dependents(self):
        g = toy_sdg()
        tracker = FrontierTracker(g)
        # op1 (needs q12, q5 resolved) and leaf quantities are available;
        # op2 needs sub + q4; sub needs op1; goal needs op2
        frontier = set(tracker.frontier())
        self.assertIn("q12", frontier)
        self.assertIn("q5", frontier)
        self.assertIn("q4", frontier)
        self.assertNotIn("op1", frontier)      # prerequisites unresolved
        self.assertNotIn("sub", frontier)      # derived from op1
        self.assertNotIn("op2", frontier)
        self.assertNotIn("goal", frontier)

    def test_resolution_unlocks_dependents(self):
        g = toy_sdg()
        tracker = FrontierTracker(g, resolve_gate=lambda q, n: True)
        tracker.step("q?", "q12")
        tracker.step("q?", "q5")
        self.assertIn("op1", tracker.frontier())
        self.assertNotIn("op1", tracker.state.resolved)

    def test_failed_gate_keeps_node_unresolved(self):
        g = toy_sdg()
        tracker = FrontierTracker(g, resolve_gate=lambda q, n: False)
        tracker.step("q?", "q12")
        self.assertNotIn("q12", tracker.state.resolved)
        # still in frontier
        self.assertIn("q12", tracker.frontier())

    def test_terminal_condition(self):
        g = toy_sdg()
        tracker = FrontierTracker(g, resolve_gate=lambda q, n: True)
        for nid in ["q12", "q5", "op1", "sub", "q4", "op2", "goal"]:
            tracker.step("q?", nid)
        self.assertEqual(tracker.frontier(), [])


class TestSelector(unittest.TestCase):
    def test_goal_wards_progress(self):
        """With everything resolved except two options, the one closer to the
        goal (larger depth) should be selected when confidences are equal."""
        g = toy_sdg()
        sel = select_frontier_node(g, ["q12", "op1"], resolved=["q5"])
        self.assertEqual(sel, "op1")   # deeper + closer to goal

    def test_redundancy_penalty(self):
        import torch
        g = toy_sdg()
        emb = {"q12": torch.tensor([1.0, 0.0]), "q5": torch.tensor([1.0, 0.0])}
        # q5 is nearly identical to resolved q12 -> penalized
        sel = select_frontier_node(g, ["q12", "q5"], resolved=["q12"], node_emb=emb)
        self.assertEqual(sel, "q12")

    def test_empty_frontier(self):
        g = toy_sdg()
        self.assertIsNone(select_frontier_node(g, [], resolved=[]))


class TestActionFrame(unittest.TestCase):
    def test_serialization_contains_fields(self):
        g = toy_sdg()
        frame = build_action_frame(g, "op1", ["q12"])
        s = serialize_frame(frame)
        self.assertIn("Target:", s)
        self.assertIn("Intent:", s)
        self.assertIn("select-operation", s)
        self.assertIn("Prerequisites:", s)
        self.assertIn("Avoid:", s)
        self.assertTrue(s.endswith(
            "Ask exactly one Socratic question that helps derive the target "
            "without directly revealing the answer."))


if __name__ == "__main__":
    unittest.main()
