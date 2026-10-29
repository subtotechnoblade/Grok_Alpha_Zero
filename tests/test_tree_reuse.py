"""Tree reuse after prune_tree: the kept subtree hangs off the new root. python -m unittest tests.test_tree_reuse"""
import unittest

import numpy as np

from Gomoku.game import Gomoku
from MCTS import MCTS
from tests.fake_net import FakeSession
from tests.test_virtual_loss import all_nodes


def make(game, k):
    np.random.seed(0)
    session = FakeSession(game.get_input_state().shape, game.policy_shape[0], seed=0)
    return MCTS(game, session, c_puct_init=2.0, fpu_reduction=0.2, use_dirichlet=True, dirichlet_alpha=0.05,
                tau=1.0, leaves_per_batch=k)


def advance(game, mcts, sims, moves):
    for _ in range(moves):
        move, _ = mcts.run(iteration_limit=sims, use_bar=False)
        game.do_action(move)
        mcts.prune_tree(move)


class TreeReuseTest(unittest.TestCase):
    def test_every_node_reaches_the_root(self):
        game = Gomoku()
        mcts = make(game, 8)
        advance(game, mcts, 400, 6)
        for node in all_nodes(mcts.root):
            top = node
            while top.parent is not None:
                top = top.parent
            self.assertIs(top, mcts.root)

    def test_root_children_point_at_the_root(self):
        game = Gomoku()
        mcts = make(game, 8)
        advance(game, mcts, 400, 4)
        self.assertTrue(mcts.root.children)
        for child in mcts.root.children:
            self.assertIs(child.parent, mcts.root)

    def test_virtual_loss_lands_on_the_root_after_reuse(self):
        game = Gomoku()
        mcts = make(game, 8)
        advance(game, mcts, 400, 4)
        leaf = max(mcts.root.children, key=lambda c: len(c.children))
        while leaf.children:
            leaf = leaf.children[0]
        mcts._apply_virtual_loss(leaf)
        self.assertEqual(mcts.root.inflight_total, 1)
        mcts._revert_virtual_loss(leaf)
        self.assertEqual(mcts.root.inflight_total, 0)


if __name__ == "__main__":
    unittest.main()
