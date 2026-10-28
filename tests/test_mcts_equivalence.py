"""R1: the live MCTS must reproduce the frozen pre-virtual-loss behaviour bit for bit.
python -m unittest tests.test_mcts_equivalence"""
import os
import pickle
import unittest

import numpy as np

from tests.mcts_scenarios import ROOT, SCENARIOS, run_scenario
from MCTS import MCTS

with open(os.path.join(ROOT, "tests", "data", "mcts_reference.pkl"), "rb") as f:
    REFERENCE = pickle.load(f)


def assert_same(test, expected, actual, where):
    test.assertEqual(len(expected["records"]), len(actual["records"]), f"{where}: move count")
    for i, (e, a) in enumerate(zip(expected["records"], actual["records"])):
        at = f"{where} move {i}"
        for key in ("move", "child_visits", "child_values", "child_prob_priors"):
            test.assertTrue(np.array_equal(e[key], a[key]), f"{at}: {key}\n{e[key]}\n{a[key]}")
            test.assertEqual(e[key].dtype, a[key].dtype, f"{at}: {key} dtype")
        test.assertEqual(e["root_visits"], a["root_visits"], f"{at}: root visits")
        test.assertEqual(len(e["move_probs"]), len(a["move_probs"]), f"{at}: move_probs length")
        for row_e, row_a in zip(e["move_probs"], a["move_probs"]):
            test.assertTrue(np.array_equal(row_e[0], row_a[0]), f"{at}: move_probs action")
            test.assertEqual(row_e[1:], row_a[1:], f"{at}: move_probs stats")
    test.assertEqual(expected["cache_stats"], actual["cache_stats"], f"{where}: cache stats")


class TestEquivalence(unittest.TestCase):
    def test_scenarios(self):
        for name in SCENARIOS:
            with self.subTest(name):
                assert_same(self, REFERENCE[name], run_scenario(MCTS, name), name)

    def test_repeatable(self):
        first = run_scenario(MCTS, "gomoku_fake_tau1")
        assert_same(self, first, run_scenario(MCTS, "gomoku_fake_tau1"), "rerun")


if __name__ == "__main__":
    unittest.main()
