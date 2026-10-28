"""Virtual loss bookkeeping (R2, R3). python -m unittest tests.test_virtual_loss"""
import unittest

import numpy as np

from Gomoku.game import Gomoku
from MCTS import MCTS
from tests.fake_net import FakeSession


def build_tree(sims=300, seed=0):
    np.random.seed(seed)
    game = Gomoku()
    game.do_action((7, 7))
    session = FakeSession(game.get_input_state().shape, game.policy_shape[0])
    mcts = MCTS(game, session, c_puct_init=2.0, fpu_reduction=0.2, use_dirichlet=False, tau=1.0)
    mcts.run(iteration_limit=sims, use_bar=False)
    return mcts


def all_nodes(root):
    stack, out = [root], []
    while stack:
        node = stack.pop()
        out.append(node)
        stack.extend(getattr(node, "children", []))
    return out


def snapshot(root):
    snap = []
    for node in all_nodes(root):
        snap.append(tuple(np.array(getattr(node, name)).copy() if hasattr(node, name) else None
                          for name in ("child_visits", "child_values", "child_prob_priors")))
    return snap


def deepest_leaf(root, depth):
    node = root
    for _ in range(depth):
        expanded = [c for c in node.children if c.is_terminal is None and c.children]
        node = max(expanded, key=lambda c: len(c.children))
    return node


def root_scores(mcts, kernel_inflight):
    root = mcts.root
    args = (root.child_prob_priors, root.child_values, root.child_visits)
    tail = (mcts.c_puct_init, mcts.c_puct_base, mcts.fpu_reduction)
    if kernel_inflight is None:
        return mcts._get_best_PUCT_score_index(*args, root.visits, *tail)
    return mcts._get_best_PUCT_score_index_inflight(*args, kernel_inflight, root.visits + int(kernel_inflight.sum()),
                                                    *tail, mcts.virtual_loss)


class TestKernel(unittest.TestCase):
    def test_zero_inflight_matches_original(self):
        rng = np.random.default_rng(0)
        for trial in range(3000):
            n = int(rng.integers(1, 226))
            priors = rng.dirichlet(np.ones(n)).astype(np.float32)
            visits = rng.integers(0, 50, n).astype(np.int64) * (rng.random(n) < 0.6)
            values = (rng.uniform(-1, 1, n) * visits).astype(np.float64)
            parent = int(visits.sum()) + int(rng.integers(0, 5))
            fpu = float(rng.choice([0.0, 0.2]))
            a = MCTS._get_best_PUCT_score_index(priors, values, visits, parent, 2.0, 19652.0, fpu)
            b = MCTS._get_best_PUCT_score_index_inflight(priors, values, visits, np.zeros(n, np.int64), parent,
                                                         2.0, 19652.0, fpu, 1.0)
            self.assertEqual(a, b, f"trial {trial}")


class TestApplyRevert(unittest.TestCase):
    def test_virtual_loss_moves_selection_away(self):
        mcts = build_tree()
        best = root_scores(mcts, None)
        child = mcts.root.children[best]
        moved = False
        for _ in range(64):
            mcts._apply_virtual_loss(child)
            if root_scores(mcts, mcts.root.child_inflight) != best:
                moved = True
                break
        self.assertTrue(moved, "virtual loss never moved the root choice")

    def test_apply_revert_is_identity(self):
        mcts = build_tree()
        before = snapshot(mcts.root)
        leaves = [deepest_leaf(mcts.root, d) for d in (1, 2, 3)] + list(mcts.root.children[:5])
        for leaf in leaves * 3:
            mcts._apply_virtual_loss(leaf)
        self.assertGreater(mcts.root.inflight_total, 0)
        for leaf in reversed(leaves * 3):
            mcts._revert_virtual_loss(leaf)
        after = snapshot(mcts.root)
        self.assertEqual(len(before), len(after))
        for b, a in zip(before, after):
            for x, y in zip(b, a):
                if x is None:
                    self.assertIsNone(y)
                else:
                    self.assertTrue(np.array_equal(x, y))
                    self.assertEqual(x.dtype, y.dtype)
        for node in all_nodes(mcts.root):
            self.assertIsNone(node.child_inflight)
            self.assertEqual(node.inflight_total, 0)

    def test_counts_along_path(self):
        mcts = build_tree()
        leaf = deepest_leaf(mcts.root, 3)
        mcts._apply_virtual_loss(leaf)
        mcts._apply_virtual_loss(leaf)
        node = leaf
        while node.parent is not None:
            parent = node.parent
            self.assertEqual(parent.child_inflight[node.child_id], 2)
            self.assertEqual(parent.inflight_total, 2)
            self.assertEqual(int(parent.child_inflight.sum()), 2)
            node = parent

    def test_virtual_loss_lowers_q_for_each_mover(self):
        mcts = build_tree()
        grandchild = deepest_leaf(mcts.root, 2)
        middle = grandchild.parent
        q_before = middle.child_values[grandchild.child_id] / middle.child_visits[grandchild.child_id]
        mcts._apply_virtual_loss(grandchild)
        idx = grandchild.child_id
        n = middle.child_visits[idx] + middle.child_inflight[idx]
        q_after = (middle.child_values[idx] - mcts.virtual_loss * middle.child_inflight[idx]) / n
        self.assertLess(q_after, q_before)
        root_idx = middle.child_id
        n_root = mcts.root.child_visits[root_idx] + mcts.root.child_inflight[root_idx]
        q_root_after = (mcts.root.child_values[root_idx] - mcts.virtual_loss * mcts.root.child_inflight[root_idx]) / n_root
        q_root_before = mcts.root.child_values[root_idx] / mcts.root.child_visits[root_idx]
        self.assertLess(q_root_after, q_root_before)

    def test_zero_virtual_loss_only_adds_visits(self):
        mcts = build_tree()
        mcts.update_hyperparams(virtual_loss=0.0)
        child = mcts.root.children[0]
        mcts._apply_virtual_loss(child)
        values = mcts.root.child_values.astype(np.float32) - 0.0 * mcts.root.child_inflight.astype(np.float32)
        self.assertTrue(np.array_equal(values, mcts.root.child_values.astype(np.float32)))


if __name__ == "__main__":
    unittest.main()
