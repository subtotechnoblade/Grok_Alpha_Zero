"""Batched search under virtual loss (R2, R4, R5, R6, R8). python -m unittest tests.test_batched_search"""
import time
import unittest

import numpy as np

from Connect4.game import Connect4
from Gomoku.game import Gomoku
from MCTS import MCTS
from Session_Cache import Cache_Wrapper
from tests.fake_net import FakeSession
from tests.mcts_scenarios import FORCED_BLOCK, IMMEDIATE_WIN
from tests.test_virtual_loss import all_nodes


class RecordingMCTS(MCTS):
    """Records every (parent, action) reserved per network batch."""

    def __init__(self, *args, **kwargs):
        self.batches = []
        self._current = []
        super().__init__(*args, **kwargs)

    def _reserve(self, node):
        leaf, payload, pending = super()._reserve(node)
        if pending:
            self._current.append((id(node), leaf.action_history[-1].tobytes()))
        return leaf, payload, pending

    def _evaluate(self, payloads):
        self.batches.append(self._current)
        self._current = []
        return super()._evaluate(payloads)


def make(game, k, session="fake", seed=0, scale=0.3, fpu=0.2, **kwargs):
    np.random.seed(seed)
    if session == "fake":
        session = FakeSession(game.get_input_state().shape, game.policy_shape[0], scale=scale)
    elif session == "cache":
        session = Cache_Wrapper(FakeSession(game.get_input_state().shape, game.policy_shape[0]), 4)
    return RecordingMCTS(game, session, c_puct_init=2.0, fpu_reduction=fpu, use_dirichlet=True,
                         dirichlet_alpha=0.3, tau=1.0, leaves_per_batch=k, **kwargs)


def assert_clean(test, mcts):
    for node in all_nodes(mcts.root):
        test.assertIsNone(node.child_inflight)
        test.assertEqual(node.inflight_total, 0)
        if node.is_terminal is None and node.children:
            for child in node.children:
                test.assertIsNotNone(child.child_prob_priors if child.is_terminal is None else 0, "pending child left")


def assert_no_duplicates(test, mcts):
    for batch in mcts.batches:
        test.assertEqual(len(batch), len(set(batch)), "same (parent, action) twice in one batch")


def play(test, game, mcts, sims, moves):
    for _ in range(moves):
        if game.action_history and game.check_win() != -2:
            break
        visits_before = mcts.root.visits
        child_before = int(np.sum(mcts.root.child_visits))
        move, probs = mcts.run(iteration_limit=sims, use_bar=False)
        n_legal = len(game.get_legal_actions())
        expected = 1 if n_legal == 1 else (3 * n_legal if sims < n_legal else sims)
        test.assertEqual(mcts.root.visits - visits_before, expected, "root visits must grow by exactly the sims")
        test.assertEqual(int(np.sum(mcts.root.child_visits)) - child_before, expected)
        test.assertAlmostEqual(sum(p[1] for p in probs), 1.0, places=6)
        assert_clean(test, mcts)
        assert_no_duplicates(test, mcts)
        game.do_action(move)
        mcts.prune_tree(move)


class TestBatchedSearch(unittest.TestCase):
    def test_gomoku_full_batches(self):
        for k in (2, 8, 32):
            with self.subTest(k=k):
                game = Gomoku()
                game.do_action((7, 7))
                mcts = make(game, k, scale=0.01, fpu=0.0)  # flat priors, no FPU: a wide tree
                play(self, game, mcts, sims=400, moves=4)
                rows = mcts.search_stats["network_rows"]
                self.assertGreater(rows / mcts.search_stats["batches"], 0.8 * k, mcts.search_stats)
                self.assertEqual(mcts.session.rows, rows + 1)  # + the first root expansion

    def test_sharp_policy_collides_but_stays_exact(self):
        game = Gomoku()
        game.do_action((7, 7))
        mcts = make(game, 32, scale=0.3)
        play(self, game, mcts, sims=400, moves=4)
        self.assertGreater(mcts.search_stats["collisions"], 0)

    def test_connect4_whole_games(self):
        for k, seed in ((4, 1), (16, 2), (64, 3)):
            with self.subTest(k=k):
                game = Connect4()
                mcts = make(game, k, seed=seed)
                play(self, game, mcts, sims=250, moves=42)

    def test_collisions_on_a_narrow_tree(self):
        game = Connect4()
        mcts = make(game, 64)
        play(self, game, mcts, sims=300, moves=3)
        self.assertGreater(mcts.search_stats["collisions"], 0)

    def test_uneven_last_batch(self):
        game = Gomoku()
        mcts = make(game, 16)
        play(self, game, mcts, sims=37, moves=3)
        self.assertTrue(all(len(b) <= 16 for b in mcts.batches))

    def test_gen0_batched(self):
        game = Gomoku()
        mcts = make(game, 8, session=None)
        play(self, game, mcts, sims=300, moves=3)

    def test_cache_batched(self):
        game = Gomoku()
        mcts = make(game, 8, session="cache")
        play(self, game, mcts, sims=300, moves=3)
        stats = mcts.session.stats()
        self.assertGreater(stats["hits"] + stats["misses"], 0)
        self.assertLessEqual(stats["entries"], stats["misses"])  # a position twice in one batch is two misses

    def test_terminal_positions(self):
        for opening, expected in ((FORCED_BLOCK, (11, 7)), (IMMEDIATE_WIN, (11, 7))):
            with self.subTest(opening=len(opening)):
                game = Gomoku()
                for action in opening:
                    game.do_action(action)
                mcts = make(game, 8)
                mcts.update_hyperparams(tau=0.0)
                move, _ = mcts.run(iteration_limit=400, use_bar=False)
                self.assertEqual(tuple(int(v) for v in move), expected)
                assert_clean(self, mcts)

    def test_time_limit(self):
        game = Gomoku()
        mcts = make(game, 8)
        start = time.time()
        mcts.run(time_limit=0.5, use_bar=False)
        self.assertLess(time.time() - start, 3.0)
        self.assertGreater(mcts.root.visits, 8)
        assert_clean(self, mcts)

    def test_new_root_mode(self):
        game = Gomoku()
        mcts = make(game, 8)
        for _ in range(3):
            move, _ = mcts.run(iteration_limit=200, use_bar=False)
            game.do_action(move)
            mcts.prune_tree(move, create_new_root=True)
            assert_clean(self, mcts)


class TestCacheBatch(unittest.TestCase):
    def test_rows_match_direct_and_counts(self):
        game = Gomoku()
        fake = FakeSession(game.get_input_state().shape, 225, seed=3)
        cache = Cache_Wrapper(FakeSession(game.get_input_state().shape, 225, seed=3), max_cache_depth=3)
        rng = np.random.default_rng(0)
        pool = rng.integers(-1, 2, (6, 15, 15, 4)).astype(np.float32)
        hits = misses = 0
        seen = set()
        for _ in range(40):
            idx = rng.integers(0, 6, int(rng.integers(1, 9)))
            depths = rng.integers(0, 6, len(idx))
            batch = pool[idx]
            policy, value = cache.run_batch(["policy", "value"], batch, depths)
            direct_p, direct_v = fake.run(["policy", "value"], {"inputs": batch})
            self.assertTrue(np.array_equal(policy, direct_p))
            self.assertTrue(np.array_equal(value, direct_v))
            seen_before = set(seen)
            for i, d in zip(idx, depths):
                if d < 3:
                    if i in seen_before:
                        hits += 1
                    else:
                        misses += 1
                        seen.add(i)
        self.assertEqual(cache.hits, hits)
        self.assertEqual(cache.misses, misses)
        self.assertEqual(len(cache.cache), len(seen))


if __name__ == "__main__":
    unittest.main()
