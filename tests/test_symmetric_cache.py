"""Symmetry-averaged and random-symmetry cache evals. python -m unittest tests.test_symmetric_cache"""
import unittest

import numpy as np

from Gomoku.game import Gomoku
from MCTS import MCTS
from Session_Cache import Cache_Wrapper
from tests.fake_net import FakeSession

ALL = np.arange(8)


def random_states(rng, n):
    states = []
    for _ in range(n):
        game = Gomoku()
        for index in rng.choice(225, int(rng.integers(0, 6)), replace=False):
            game.do_action((int(index) % 15, int(index) // 15))
        states.append(game.get_input_state())
    return np.array(states)


class CountingSession(FakeSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_sizes = []

    def run(self, output_names, input_feed, run_options=None):
        self.batch_sizes.append(len(input_feed["inputs"]))
        return super().run(output_names, input_feed, run_options)


class TestTransforms(unittest.TestCase):
    def test_undo_inverts_apply(self):
        rng = np.random.default_rng(0)
        policy = rng.random((16, 225)).astype(np.float32)
        ids = np.tile(ALL, 2)
        moved = Gomoku.apply_symmetry(policy.reshape(16, 15, 15, 1), ids).reshape(16, 225)
        np.testing.assert_array_equal(Gomoku.undo_symmetry(moved, ids), policy)

    def test_eight_distinct_symmetries(self):
        game = Gomoku()
        game.do_action((2, 1))
        moved = Gomoku.apply_symmetry(np.repeat(game.get_input_state()[None], 8, 0), ALL)
        stones = {tuple(np.argwhere(s[..., 1] == 1)[0]) for s in moved}
        self.assertEqual(stones, {(1, 2), (2, 1), (1, 12), (12, 1), (2, 13), (13, 2), (12, 13), (13, 12)})

    def test_stone_and_policy_move_together(self):
        state = np.zeros((8, 15, 15, 4), np.float32)
        state[:, 3, 5, 0] = 1
        policy = np.zeros((8, 225), np.float32)
        policy[:, 3 * 15 + 5] = 1
        moved_states = Gomoku.apply_symmetry(state, ALL)
        moved_policy = Gomoku.apply_symmetry(policy.reshape(8, 15, 15, 1), ALL)
        for s, p in zip(moved_states, moved_policy):
            self.assertTrue(np.array_equal(np.argwhere(s[..., 0]), np.argwhere(p[..., 0])))


class TestSymmetricCache(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.fake = FakeSession((15, 15, 4), 225, seed=2)

    def wrapper(self, session=None, **kwargs):
        return Cache_Wrapper(session or FakeSession((15, 15, 4), 225, seed=2), 3, symmetries=Gomoku, **kwargs)

    def manual_average(self, states):
        n = len(states)
        ids = np.repeat(ALL, n)
        p, v = self.fake.run(["policy", "value"], {"inputs": Gomoku.apply_symmetry(np.tile(states, (8, 1, 1, 1)), ids)})
        return Gomoku.undo_symmetry(p, ids).reshape(8, n, 225).mean(0), v.reshape(8, n, 1).mean(0)

    def test_shallow_rows_averaged_deep_rows_plain(self):
        cache = self.wrapper()
        states = random_states(self.rng, 6)
        depths = np.array([0, 5, 1, 7, 2, 3])
        policy, value = cache.run_batch(["policy", "value"], states, depths)
        shallow = depths < 3
        expected_p, expected_v = self.manual_average(states[shallow])
        np.testing.assert_allclose(policy[shallow], expected_p, rtol=1e-5)
        np.testing.assert_allclose(value[shallow], expected_v, rtol=1e-5)
        direct_p, direct_v = self.fake.run(["policy", "value"], {"inputs": states[~shallow]})
        self.assertTrue(np.array_equal(policy[~shallow], direct_p))
        self.assertTrue(np.array_equal(value[~shallow], direct_v))

    def test_random_symmetry_deep_rows(self):
        cache = self.wrapper(random_symmetries=True)
        state = random_states(self.rng, 1)
        np.random.seed(0)
        seen = set()
        for _ in range(40):
            policy, value = cache.run_batch(["policy", "value"], state, [9])
            seen.add(np.round(value[0, 0], 6))
            self.assertAlmostEqual(float(policy.sum()), 1.0, places=5)
        self.assertGreater(len(seen), 1)
        candidates = []
        for s in ALL:
            p, v = self.fake.run(["policy", "value"], {"inputs": Gomoku.apply_symmetry(state, np.array([s]))})
            candidates.append((Gomoku.undo_symmetry(p, np.array([s])), v))
        np.random.seed(5)
        policy, value = cache.run_batch(["policy", "value"], state, [9])
        self.assertTrue(any(np.allclose(policy, p) and np.allclose(value, v) for p, v in candidates))

    def test_hits_return_the_averaged_entry(self):
        cache = self.wrapper()
        states = random_states(self.rng, 3)
        first = cache.run_batch(["policy", "value"], states, [0, 1, 2])
        second = cache.run_batch(["policy", "value"], states[[2, 0]], [2, 0])
        self.assertTrue(np.array_equal(second[0], first[0][[2, 0]]))
        self.assertEqual((cache.hits, cache.misses), (2, 3))

    def test_symmetric_position_policy_is_symmetric(self):
        cache = self.wrapper()
        policy, _ = cache.run(["policy", "value"], {"inputs": Gomoku().get_input_state()[None]}, 0)
        grid = policy[0].reshape(15, 15)
        np.testing.assert_allclose(grid, grid.T, rtol=1e-5)
        np.testing.assert_allclose(grid, np.rot90(grid), rtol=1e-5)

    def test_max_rows_chunks_session_calls(self):
        session = CountingSession((15, 15, 4), 225, seed=2)
        cache = self.wrapper(session, random_symmetries=True, max_rows=8)
        states = random_states(self.rng, 5)
        policy, _ = cache.run_batch(["policy", "value"], states, [0, 1, 9, 2, 0])
        self.assertLessEqual(max(session.batch_sizes), 8)
        self.assertEqual(sum(session.batch_sizes), 4 * 8 + 1)
        expected_p, _ = self.manual_average(states[[0, 1, 3, 4]])
        np.testing.assert_allclose(policy[[0, 1, 3, 4]], expected_p, rtol=1e-5)

    def test_off_by_default_is_bit_identical(self):
        cache = Cache_Wrapper(FakeSession((15, 15, 4), 225, seed=2), 3)
        states = random_states(self.rng, 4)
        policy, value = cache.run_batch(["policy", "value"], states, [0, 1, 5, 9])
        direct_p, direct_v = self.fake.run(["policy", "value"], {"inputs": states})
        self.assertTrue(np.array_equal(policy, direct_p) and np.array_equal(value, direct_v))

    def test_search_runs_with_symmetries(self):
        np.random.seed(0)
        cache = self.wrapper(random_symmetries=True, max_rows=8)
        mcts = MCTS(Gomoku(), cache, c_puct_init=2.0, fpu_reduction=0.2, use_dirichlet=True, dirichlet_alpha=0.05,
                    tau=1.0, leaves_per_batch=8)
        move, probs = mcts.run(iteration_limit=400, use_bar=False)
        self.assertAlmostEqual(sum(p[1] for p in probs), 1.0, places=5)
        self.assertGreater(cache.misses, 0)


if __name__ == "__main__":
    unittest.main()
