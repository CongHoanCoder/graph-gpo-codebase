"""Golden tests for the decoupled advantage, curriculum, and losses.

The primary golden case is the worked example for reward dynamics during
training:

  8 rollouts, mu_rel=0.82, sigma_rel=0.07; rollout rewards (0.88, 0.91, 1.0);
  A_rel = 0.86, A_coh = 0.60, A_form = 0 (saturated);
  omega_form(250)=0.16, omega_rel=omega_coh(250)=0.42;
  hat{A} = 0.42*0.86 + 0.42*0.60 + 0.16*0.0 = 0.61
"""
import unittest

import numpy as np
import torch

from graph_cpo.advantage import (anchor_alpha, compute_advantage,
                                 curriculum_weights, scalar_grpo_advantage)
from graph_cpo.losses import (clipped_surrogate, graph_cpo_loss, kl_to_reference,
                              total_loss)


class TestCurriculum(unittest.TestCase):
    def test_curriculum_schedule(self):
        # u=25: omega_form ~ 0.65 ; u=250: ~0.16 (worked example)
        self.assertAlmostEqual(curriculum_weights(25)["form"], 0.65, places=1)
        self.assertAlmostEqual(curriculum_weights(250)["form"], 0.16, places=1)

    def test_rest_split_evenly(self):
        w = curriculum_weights(0)
        self.assertAlmostEqual(w["rel"] + w["coh"] + w["form"], 1.0, places=6)
        self.assertAlmostEqual(w["rel"], w["coh"])

    def test_monotone_decay(self):
        forms = [curriculum_weights(u)["form"] for u in range(0, 500, 50)]
        self.assertEqual(forms, sorted(forms, reverse=True))

    def test_anchor_schedule(self):
        self.assertAlmostEqual(anchor_alpha(0), 0.8, places=6)
        self.assertGreater(anchor_alpha(100), anchor_alpha(1000))
        self.assertGreaterEqual(anchor_alpha(100000), 0.05)


class TestDecoupledAdvantage(unittest.TestCase):
    @staticmethod
    def _make_rewards(v0: float, mu: float, sigma: float, m: int) -> "np.ndarray":
        """M rewards where rollout 0 = v0 and the group has exact mean/std."""
        dev0 = v0 - mu
        # remaining M-1 deviations: M-2 equal 'a' + one balancing value
        # solve 6a^2-style quadratic generically: k*a^2 + (d_last with sum constraint)
        k = m - 2
        # sum: k*a + d_last = -dev0 ; sumsq: k*a^2 + d_last^2 = m*sigma^2 - dev0^2
        # => (k + k^2) a^2 + 2*k*dev0*a + dev0^2 - (m*sigma^2 - dev0^2) = 0
        target_sumsq = m * sigma ** 2 - dev0 ** 2
        A = k + k * k
        B = 2 * k * dev0
        C = dev0 ** 2 - target_sumsq
        a = (-B + (B * B - 4 * A * C) ** 0.5) / (2 * A)
        d_last = -dev0 - k * a
        vals = [v0] + [mu + a] * k + [mu + d_last]
        arr = np.array(vals)
        assert abs(arr.mean() - mu) < 1e-9
        assert abs(arr.std() - sigma) < 1e-9
        return arr

    def test_golden_worked_example(self):
        M, T = 8, 1
        # golden example: mu_rel=0.82, sigma_rel=0.07, rollout rel=0.88 -> A=0.86
        #                 mu_coh=0.88, sigma_coh=0.05, rollout coh=0.91 -> A=0.60
        rel = self._make_rewards(0.88, 0.82, 0.07, M).reshape(M, T)
        coh = self._make_rewards(0.91, 0.88, 0.05, M).reshape(M, T)
        form = np.ones((M, T))     # all rollouts format-perfect

        adv = compute_advantage({"rel": rel, "coh": coh, "form": form}, step=250,
                                eps=1e-8)
        w = adv.weights
        self.assertAlmostEqual(w["form"], 0.16, places=1)
        self.assertAlmostEqual(w["rel"], 0.42, places=1)
        a_rel = adv.per_dim["rel"][0, 0]
        a_coh = adv.per_dim["coh"][0, 0]
        self.assertAlmostEqual(a_rel, 0.86, places=2)
        self.assertAlmostEqual(a_coh, 0.60, places=2)
        # A_form is exactly zero: every rollout format-perfect => no signal
        self.assertEqual(adv.per_dim["form"][0, 0], 0.0)
        expected = w["rel"] * a_rel + w["coh"] * a_coh
        self.assertAlmostEqual(float(adv.advantage[0, 0]), float(expected), places=6)
        # the expected arithmetic: 0.42*0.86 + 0.42*0.60 ~ 0.61
        self.assertAlmostEqual(float(expected), 0.61, delta=0.05)

    def test_saturated_dimension_contributes_nothing(self):
        M, T = 4, 3
        ones = np.ones((M, T))
        adv = compute_advantage({"rel": np.random.rand(M, T),
                                 "coh": np.random.rand(M, T),
                                 "form": ones}, step=0)
        np.testing.assert_allclose(adv.per_dim["form"], 0.0)

    def test_zero_variance_no_division_by_zero(self):
        M, T = 3, 2
        same = np.full((M, T), 0.5)
        adv = compute_advantage({"rel": same, "coh": same, "form": same}, step=0)
        np.testing.assert_allclose(adv.advantage, 0.0)

    def test_use_dims_ablation(self):
        M, T = 4, 1
        rng = np.random.RandomState(1)
        rewards = {"rel": rng.rand(M, T), "coh": rng.rand(M, T), "form": rng.rand(M, T)}
        adv = compute_advantage(rewards, step=0, use_dims=["rel"])
        w = curriculum_weights(0)
        # composite = omega_rel * A_rel exactly (other dims excluded)
        np.testing.assert_allclose(adv.advantage, w["rel"] * adv.per_dim["rel"],
                                   atol=1e-6)

    def test_nan_padding_ignored(self):
        rewards = {"rel": np.array([[0.9, np.nan], [0.1, np.nan]]),
                   "coh": np.array([[0.5, np.nan], [0.5, np.nan]]),
                   "form": np.array([[1.0, np.nan], [1.0, np.nan]])}
        adv = compute_advantage(rewards, step=0)
        w = curriculum_weights(0)
        # rel advantage of rollout 0 is +1 (it is above the group mean)
        self.assertAlmostEqual(float(adv.advantage[0, 0]), w["rel"] * 1.0, places=5)
        self.assertEqual(float(adv.advantage[0, 1]), 0.0)   # padded -> zero


class TestScalarGRPOBaseline(unittest.TestCase):
    def test_scalarizes_before_normalizing(self):
        M, T = 4, 1
        rng = np.random.RandomState(2)
        rewards = {"rel": rng.rand(M, T), "coh": rng.rand(M, T), "form": rng.rand(M, T)}
        adv = scalar_grpo_advantage(rewards, step=0)
        # composite = mean of the three; normalized to unit-ish scale
        self.assertEqual(adv.shape, (M, T))
        self.assertAlmostEqual(float(adv.mean()), 0.0, places=5)


class TestLosses(unittest.TestCase):
    def test_clipped_surrogate_form(self):
        logp_new = torch.tensor([[0.0, 0.0]])
        logp_old = torch.tensor([[0.0, 0.0]])         # ratio = 1 (no clipping)
        adv = torch.tensor([[1.0, -1.0]])
        mask = torch.ones_like(adv)
        loss = clipped_surrogate(logp_new, logp_old, adv, mask, clip_eps=0.2)
        self.assertAlmostEqual(float(loss), 0.0, places=6)   # mean of -adv = 0

    def test_clipping_active(self):
        logp_new = torch.tensor([[2.0]])              # ratio = e^2 >> 1.2
        logp_old = torch.tensor([[0.0]])
        adv = torch.tensor([[1.0]])
        mask = torch.ones_like(adv)
        loss = float(clipped_surrogate(logp_new, logp_old, adv, mask, clip_eps=0.2))
        self.assertAlmostEqual(loss, -1.2, places=5)  # clipped ratio

    def test_kl_nonnegative(self):
        torch.manual_seed(0)
        a = torch.randn(2, 5)
        b = torch.randn(2, 5)
        mask = torch.ones_like(a)
        # use log-probs: normalize rows
        a = torch.log_softmax(a, dim=-1)
        b = torch.log_softmax(b, dim=-1)
        kl = float(kl_to_reference(a, b, mask))
        self.assertGreaterEqual(kl, 0.0)

    def test_alpha_mixing(self):
        l_t = torch.tensor(2.0)
        l_c = torch.tensor(1.0)
        self.assertAlmostEqual(float(total_loss(l_t, l_c, alpha=1.0)), 2.0, places=6)
        self.assertAlmostEqual(float(total_loss(l_t, l_c, alpha=0.0)), 1.0, places=6)
        self.assertAlmostEqual(float(total_loss(None, l_c, alpha=0.5)), 1.0, places=6)

    def test_graph_cpo_loss_components(self):
        logp = torch.zeros(1, 4)
        adv = torch.ones(1, 4)
        mask = torch.ones(1, 4)
        out = graph_cpo_loss(logp, logp, logp, adv, mask, clip_eps=0.2, kl_beta=0.01)
        self.assertAlmostEqual(float(out["loss"]), -1.0, places=5)
        self.assertAlmostEqual(float(out["kl_ref"]), 0.0, places=5)
        self.assertAlmostEqual(float(out["clip_frac"]), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
