"""Playout cap decisions. python -m unittest tests.test_playout_cap"""
import unittest

import numpy as np

from Playout_Cap import Playout_Cap, root_signals


def config(**overrides):
    base = dict(full_search_prob=0.25, cheap_sims=128, escalate_policy=True, escalate_value=True,
                escalate_close=True, escalate_percentile=90.0, close_gap=0.02, close_min_visits=10)
    base.update(overrides)
    return base


class SignalTest(unittest.TestCase):
    def test_agreement_has_no_surprise(self):
        priors = np.array([0.5, 0.3, 0.2])
        visits = np.array([50, 30, 20])
        values = visits * 0.1
        kl, changed, value_surprise, gap = root_signals(visits, values, priors, 0.1, 10)
        self.assertAlmostEqual(kl, 0.0, places=12)
        self.assertFalse(changed)
        self.assertAlmostEqual(value_surprise, 0.0, places=12)
        self.assertAlmostEqual(gap, 0.0, places=12)

    def test_disagreement(self):
        priors = np.array([0.7, 0.2, 0.1])
        visits = np.array([10, 80, 10])
        values = np.array([10 * -0.5, 80 * 0.3, 10 * 0.0])
        kl, changed, value_surprise, gap = root_signals(visits, values, priors, -0.4, 10)
        self.assertGreater(kl, 0.5)
        self.assertTrue(changed)
        self.assertAlmostEqual(value_surprise, abs((values.sum() / 100) + 0.4))
        self.assertAlmostEqual(gap, 0.3)

    def test_close_ignores_thin_moves(self):
        visits = np.array([100, 5, 40])
        values = np.array([100 * 0.20, 5 * 0.21, 40 * 0.05])
        self.assertAlmostEqual(root_signals(visits, values, np.ones(3) / 3, 0.0, 10)[3], 0.15)


class EscalationTest(unittest.TestCase):
    def test_rate_tracks_the_percentile(self):
        np.random.seed(0)
        cap = Playout_Cap(config(escalate_value=False, escalate_close=False))
        priors = np.array([0.4, 0.3, 0.2, 0.1])
        fired = 0
        for _ in range(3000):
            visits = np.random.multinomial(128, priors).astype(float)
            fired += cap.escalate(visits, np.zeros(4), priors) is not None
        self.assertLess(fired / 3000, 0.25)
        self.assertGreater(fired / 3000, 0.05)

    def test_triggers_are_independent(self):
        cap = Playout_Cap(config(escalate_policy=False, escalate_value=False, escalate_close=True))
        priors = np.array([0.7, 0.2, 0.1])
        self.assertIsNone(cap.escalate([10, 80, 10], [-5, 24, 0], priors))
        self.assertEqual(cap.escalate([50, 50, 0], [10, 10.5, 0], priors), "close")

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
