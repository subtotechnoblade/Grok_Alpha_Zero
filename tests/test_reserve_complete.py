"""Split expansion (R6, R7). python -m unittest tests.test_reserve_complete"""
import unittest

import numpy as np

from Gomoku.game import Gomoku
from MCTS import MCTS
from tests.fake_net import FakeSession
from tests.mcts_scenarios import FORCED_BLOCK
from tests.test_virtual_loss import all_nodes, build_tree


class TestProgressiveOrder(unittest.TestCase):
    def test_children_follow_prior_order(self):
        mcts = build_tree(sims=500)
        checked = 0
        for node in all_nodes(mcts.root):
            if node.is_terminal is not None or not node.children:
                continue
            for i, child in enumerate(node.children):
                self.assertEqual(child.child_id, i)
            priors = node.child_prob_priors
            self.assertTrue(np.all(np.diff(priors) <= 0), "priors not sorted descending")
            if hasattr(node, "child_legal_actions"):
                for i, child in enumerate(node.children):
                    self.assertTrue(np.array_equal(child.action_history[-1], node.child_legal_actions[i]))
                checked += 1
        self.assertGreater(checked, 0)


class TestPending(unittest.TestCase):
    def test_reserve_then_complete(self):
        mcts = build_tree(sims=50)
        parent = mcts.root
        n_before = len(parent.children)
        expected_action = parent.child_legal_actions[n_before]
        child, (inputs, depth), pending = mcts._reserve(parent)
        self.assertTrue(pending)
        self.assertIs(parent.children[-1], child)
        self.assertEqual(child.child_id, n_before)
        self.assertTrue(np.array_equal(child.action_history[-1], expected_action))
        self.assertIsNone(child.child_prob_priors)
        self.assertEqual(inputs.shape, mcts.game.get_input_state().shape)
        self.assertEqual(depth, len(parent.action_history))
        self.assertEqual(len(child.child_visits), len(child.child_legal_actions))

        policy, value = mcts._compute_outputs(inputs, depth)
        backed = mcts._complete(child, policy, value)
        self.assertEqual(backed, -value)
        self.assertTrue(np.all(np.diff(child.child_prob_priors) <= 0))
        self.assertEqual(len(child.child_prob_priors), len(child.child_visits))

    def test_selection_stops_at_pending_child(self):
        mcts = build_tree(sims=50)
        child, _, _ = mcts._reserve(mcts.root)
        mcts.root.child_prob_priors[child.child_id] = 1e6
        self.assertIs(mcts._PUCT_select(), child)

    def test_terminal_resolves_without_network(self):
        np.random.seed(0)
        game = Gomoku()
        for action in FORCED_BLOCK:
            game.do_action(action)
        session = FakeSession(game.get_input_state().shape, 225)
        mcts = MCTS(game, session, use_dirichlet=False, tau=0.0)
        rows_before = session.rows
        resolved = 0
        for _ in range(len(mcts.root.child_legal_actions) - len(mcts.root.children)):
            leaf, payload, pending = mcts._reserve(mcts.root)
            if not pending:
                resolved += 1
                self.assertIsNone(leaf.is_terminal)
                self.assertTrue(all(c.is_terminal is not None for c in leaf.children))
            else:
                break
        self.assertEqual(session.rows, rows_before)


if __name__ == "__main__":
    unittest.main()
