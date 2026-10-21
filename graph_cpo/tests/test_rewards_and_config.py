"""Format reward + declarative intent + config loading tests."""
import unittest

from graph_cpo.config import load_config
from graph_cpo.rewards.coherence import DeclarativeIntentExtractor
from graph_cpo.rewards.format import extract_question, format_reward


class TestFormatReward(unittest.TestCase):
    def test_valid_tagged(self):
        self.assertEqual(format_reward("<question>How many apples?</question>"), 1.0)

    def test_missing_tag(self):
        self.assertEqual(format_reward("How many apples?"), 0.0)

    def test_tag_without_question_mark(self):
        self.assertEqual(format_reward("<question>apples</question>"), 0.0)

    def test_extra_chatter_rejected(self):
        self.assertEqual(
            format_reward("Sure! <question>How many?</question> Hope that helps!"), 0.0)

    def test_fallback_lenient_mode(self):
        self.assertEqual(format_reward("How many apples?", strict=False), 1.0)
        self.assertEqual(format_reward("apples", strict=False), 0.0)

    def test_extract_question(self):
        self.assertEqual(
            extract_question("<question>How many apples?</question>"),
            "How many apples?")


class TestIntentExtractor(unittest.TestCase):
    def test_basic(self):
        d = DeclarativeIntentExtractor()
        out = d("How many cupcakes does Sarah have at the start?")
        self.assertTrue(out.startswith("The question asks for"))
        self.assertNotIn("?", out)


class TestConfigs(unittest.TestCase):
    def test_dev_config_loads(self):
        cfg = load_config("graph_cpo/configs/dev_qwen_0.5b.yaml")
        self.assertIn("Qwen", cfg.model.backbone)
        self.assertEqual(cfg.training.rl.rollouts_per_prompt, 4)
        self.assertEqual(cfg.planning.lambda_g, 0.4)

    def test_full_scale_config_values(self):
        cfg = load_config("graph_cpo/configs/paper_qwen32b.yaml")
        self.assertEqual(cfg.model.lora.r, 64)
        self.assertEqual(cfg.model.lora.alpha, 128)
        self.assertEqual(cfg.sdg.gnn.layers, 3)
        self.assertEqual(cfg.sdg.gnn.hidden, 768)
        self.assertEqual(cfg.sdg.prefix_dim, 1024)
        self.assertEqual(cfg.training.sft.epochs, 3)
        self.assertEqual(cfg.training.sft.lr, 2e-5)
        rl = cfg.training.rl
        self.assertEqual(rl.epochs, 50)
        self.assertEqual(rl.rollouts_per_prompt, 8)
        self.assertEqual(rl.kl_beta, 0.01)
        self.assertEqual(rl.clip_eps, 0.2)
        self.assertEqual(rl.curriculum_lambda, 0.01)
        self.assertEqual((rl.omega_min, rl.omega_max), (0.1, 0.8))
        self.assertEqual((rl.alpha0, rl.alpha_min, rl.alpha_lambda), (0.8, 0.05, 0.01))
        self.assertEqual(cfg.rewards.eta, 0.5)
        self.assertEqual(cfg.planning.sim_threshold, 0.75)
        self.assertEqual(cfg.planning.nli_threshold, 0.6)
        self.assertEqual(
            (cfg.planning.lambda_g, cfg.planning.lambda_d,
             cfg.planning.lambda_c, cfg.planning.lambda_r), (0.4, 0.3, 0.2, 0.1))
        self.assertEqual(cfg.eval.solver_max_tokens, 4096)
        self.assertEqual(cfg.eval.solver_timeout_s, 300)
        self.assertEqual(cfg.eval.num_decoding_runs, 16)


if __name__ == "__main__":
    unittest.main()
