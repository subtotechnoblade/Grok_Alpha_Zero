"""Playout cap decisions. python -m unittest tests.test_playout_cap"""
import unittest

import numpy as np

from Playout_Cap import Playout_Cap, policy_surprise, prior_disagreement, value_gap


def config(**overrides):
    base = dict(full_search_prob=0.25, cheap_sims=128, escalate_policy=True, escalate_value=True,
                escalate_disagreement=np.log(2), escalate_value_gap=0.3)
    base.update(overrides)
    return base


class SignalTest(unittest.TestCase):
    def test_agreement_is_zero(self):
        self.assertEqual(prior_disagreement([90, 30, 8], [0.5, 0.3, 0.2]), 0.0)

    def test_disagreement_ignores_how_flat_the_prior_is(self):
        flat = np.full(100, 0.01)
        flat[0] = 0.02
        visits = np.zeros(100)
        visits[0] = 128
        self.assertEqual(prior_disagreement(visits, flat), 0.0)
        self.assertGreater(policy_surprise(visits, flat), 3.0)

    def test_disagreement_is_the_log_prior_ratio(self):
        self.assertAlmostEqual(prior_disagreement([10, 100, 18], [0.6, 0.2, 0.2]), np.log(3))

    def test_value_gap(self):
        self.assertAlmostEqual(value_gap([50, 50], [10, 20], -0.2), 0.5)
        self.assertEqual(value_gap([0, 0], [0, 0], 0.4), 0.0)


class EscalationTest(unittest.TestCase):
    def test_reasons(self):
        cap = Playout_Cap(config())
        priors = [0.6, 0.2, 0.2]
        self.assertIsNone(cap.escalate([100, 20, 8], [10, 2, 0], priors, 0.1))
        self.assertEqual(cap.escalate([20, 100, 8], [2, 10, 0], priors, 0.1), "policy")
        self.assertEqual(cap.escalate([100, 20, 8], [60, 2, 0], priors, -0.2), "value")

    def test_triggers_are_independent(self):
        priors = [0.6, 0.2, 0.2]
        no_value = Playout_Cap(config(escalate_value=False))
        self.assertIsNone(no_value.escalate([100, 20, 8], [60, 2, 0], priors, -0.2))
        no_policy = Playout_Cap(config(escalate_policy=False))
        self.assertIsNone(no_policy.escalate([20, 100, 8], [2, 10, 0], priors, 0.1))

    def test_thresholds_are_absolute(self):
        cap = Playout_Cap(config())
        priors = [0.6, 0.2, 0.2]
        for _ in range(500):
            self.assertIsNone(cap.escalate([100, 20, 8], [10, 2, 0], priors, 0.1))
        self.assertEqual(cap.escalate([20, 100, 8], [2, 10, 0], priors, 0.1), "policy")

    def test_off_by_default(self):
        cap = Playout_Cap({})
        self.assertFalse(cap.enabled)
        self.assertIsNone(cap.escalate([10, 80, 10], [-5, 24, 0], [0.7, 0.2, 0.1], 0.9))

    def test_full_search_probability(self):
        np.random.seed(1)
        cap = Playout_Cap(config())
        rate = np.mean([cap.full_search() for _ in range(20000)])
        self.assertAlmostEqual(rate, 0.25, delta=0.01)


if __name__ == "__main__":
    unittest.main()
