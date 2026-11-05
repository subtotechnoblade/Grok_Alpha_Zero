"""MCTS draw accumulator (track_draws). python -m unittest tests.test_draw_tracking"""
import unittest

import numpy as np

from Gomoku.game import Gomoku
from MCTS import MCTS
from tests.fake_net import FakeSession
from tests.mcts_scenarios import make_session
from tests.test_full_board_draw import draw_order, play
from tests.test_virtual_loss import all_nodes


class ConstantDraw(FakeSession):
    def run(self, output_names, input_feed, run_options=None):
        outputs = super().run(output_names, input_feed, run_options)
        return [np.full_like(out, 0.3) if name == "draw" else out for name, out in zip(output_names, outputs)]


def search(track, kind="cached", k=8, sims=400, moves=3, session=None):
    game = Gomoku()
    np.random.seed(0)
    session = session or make_session(kind, game)
    mcts = MCTS(game, session, c_puct_init=2.0, fpu_reduction=0.2, use_dirichlet=True, dirichlet_alpha=0.05,
                tau=1.0, leaves_per_batch=k, track_draws=track)
    rows = []
    for _ in range(moves):
        move, probs = mcts.run(iteration_limit=sims, use_bar=False)
        rows.append(probs)
        game.do_action(move)
        mcts.prune_tree(move)
    return mcts, rows


class DrawTrackingTest(unittest.TestCase):
    def test_tracking_leaves_the_search_unchanged(self):
        for kind, k in (("fake", 1), ("cached", 8)):
            with self.subTest(session=kind, leaves_per_batch=k):
                off, rows_off = search(False, kind, k)
                on, rows_on = search(True, kind, k)
                for a, b in zip(all_nodes(off.root), all_nodes(on.root)):
                    if hasattr(a, "child_visits"):
                        np.testing.assert_array_equal(a.child_visits, b.child_visits)
                        np.testing.assert_array_equal(a.child_values, b.child_values)
                for move_off, move_on in zip(rows_off, rows_on):
                    for entry_off, entry_on in zip(move_off, move_on):
                        self.assertEqual(len(entry_off), 8)
                        self.assertEqual(len(entry_on), 9)
                        np.testing.assert_array_equal(entry_off[0], entry_on[0])
                        self.assertEqual(entry_off[1:8], entry_on[1:8])

    def test_every_visit_carries_its_leaf_draw(self):
        game = Gomoku()
        session = ConstantDraw(game.get_input_state().shape, game.policy_shape[0], seed=0)
        mcts, _ = search(True, k=8, session=session)
        for node in all_nodes(mcts.root):
            if getattr(node, "child_draws", None) is not None:
                np.testing.assert_allclose(node.child_draws, float(np.float32(0.3)) * node.child_visits, rtol=1e-12)

    def test_draw_mass_is_bounded_by_visits(self):
        mcts, rows = search(True, k=16, sims=600, moves=4)
        for node in all_nodes(mcts.root):
            if getattr(node, "child_draws", None) is not None:
                self.assertTrue(np.all(node.child_draws >= 0))
                self.assertTrue(np.all(node.child_draws <= node.child_visits + 1e-9))
        for probs in rows:
            self.assertTrue(all(0.0 <= entry[8] <= 1.0 for entry in probs))

    def test_terminal_draws_count_fully(self):
        moves = draw_order()
        for left in (1, 2):
            with self.subTest(empty_cells=left), np.errstate(all="ignore"):
                game = play(moves[:-left])
                expected = {}
                for first in game.get_legal_actions():
                    probe = play(moves[:-left])
                    probe.do_action(first)
                    while probe.check_win() == -2:
                        probe.do_action(probe.get_legal_actions()[0])
                    expected[tuple(int(v) for v in first)] = 1.0 if probe.check_win() == 0 else 0.0
                np.random.seed(0)
                mcts = MCTS(game, make_session("cached", game), c_puct_init=2.0, use_dirichlet=False, tau=0.0,
                            leaves_per_batch=1, track_draws=True)
                _, probs = mcts.run(iteration_limit=50, use_bar=False)
                for entry in probs:
                    if entry[4] > 0:
                        self.assertEqual(entry[8], expected[tuple(int(v) for v in entry[0])])


if __name__ == "__main__":
    unittest.main()
