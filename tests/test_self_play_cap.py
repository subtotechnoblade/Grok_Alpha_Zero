"""Self play with playout caps, cheap targets, surprise weights and value discount. python -m unittest tests.test_self_play_cap"""
import multiprocessing as mp
import tempfile
import unittest

import h5py as h5
import numpy as np

from MCTS import MCTS
from Quoridor import config as qc
from Quoridor.game import Quoridor
from Self_Play import SEARCH_KINDS, Self_Play
from pipeline import Make_Dataset_File
from tests.fake_net import FakeSession

SETTINGS = dict(MCTS_iteration_limit=96, cheap_sims=24, leaves_per_batch=4, max_actions=60, full_search_prob=0.25,
                escalate_policy=True, escalate_value=False, cheap_policy_targets=True,
                cheap_target_weight=0.25, surprise_weighting=True, value_discount=0.99, wdl_head=True,
                aux_heads=True, opponent_policy=True, num_explore_actions_first=2, num_explore_actions_second=2,
                endgame_solver=False)


def play_games(n, **overrides):
    folder = tempfile.mkdtemp()
    Make_Dataset_File(folder)
    tc = qc.train_config.replace(**{**SETTINGS, **overrides})
    session = FakeSession(Quoridor().get_input_state().shape, Quoridor().policy_shape[0], seed=0, scale=0.002)
    np.random.seed(0)
    recorded = [Self_Play(Quoridor(), session, qc.build_config, tc, mp.Lock(), folder, 1).play() for _ in range(n)]
    return folder, recorded


class SelfPlayCapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder, cls.recorded = play_games(4)
        with h5.File(f"{cls.folder}/Self_Play_Data.h5") as f:
            cls.stats = f["game_stats"][:]
            cls.search = f["search_stats"][:]
            cls.data = {k: f[k][:] for k in f if k not in ("game_stats", "search_stats")}

    def test_counts_add_up(self):
        moves, kinds = self.search[-2], self.search[:len(SEARCH_KINDS)]
        self.assertEqual(kinds.sum(), moves)
        self.assertEqual(self.stats[1], sum(self.recorded))
        self.assertEqual(sum(self.recorded), moves)
        self.assertGreater(kinds[SEARCH_KINDS.index("cheap")], 0)
        self.assertLess(self.search[-1] / moves, 96)

    def test_every_table_has_one_row_per_position(self):
        for i in range(2 * 4):
            n = len(self.data[f"boards_{i}"])
            for name in ("policies", "values", "aux_heads", "opponent_policies", "sample_weights"):
                self.assertEqual(len(self.data[f"{name}_{i}"]), n, name)

    def test_weights(self):
        w = np.concatenate([self.data[f"sample_weights_{i}"] for i in range(0, 8, 2)])
        self.assertTrue(np.isclose(w, 0.25).any())
        self.assertTrue((w > 0).all())
        full = w[~np.isclose(w, 0.25)]
        self.assertGreater(full.std(), 0)

    def test_targets_are_distributions(self):
        for i in range(8):
            np.testing.assert_allclose(self.data[f"policies_{i}"].sum(1), 1, atol=1e-4)
            v = self.data[f"values_{i}"]
            np.testing.assert_allclose(v.sum(1), 1, atol=1e-5)
            self.assertTrue((v >= -1e-6).all())


class RecordingTest(unittest.TestCase):
    def test_skipped_plies_break_opponent_targets(self):
        folder, recorded = play_games(2, cheap_policy_targets=False, surprise_weighting=False)
        with h5.File(f"{folder}/Self_Play_Data.h5") as f:
            counts = f["search_stats"][:]
            self.assertEqual(f["game_stats"][1], counts[-2] - counts[SEARCH_KINDS.index("cheap")])
            self.assertNotIn("sample_weights_0", f)
            opp = f["opponent_policies_0"][:]
            self.assertTrue(np.allclose(opp[-1], 0))

    def test_value_trigger_escalates(self):
        folder, _ = play_games(1, escalate_value=True, escalate_value_gap=0.0)
        with h5.File(f"{folder}/Self_Play_Data.h5") as f:
            counts = f["search_stats"][:]
        self.assertGreater(counts[SEARCH_KINDS.index("value")], 0)
        self.assertEqual(counts[SEARCH_KINDS.index("close")], 0)

    def test_discount_shrinks_early_targets(self):
        folder, _ = play_games(3, full_search_prob=1.0, cheap_policy_targets=False, surprise_weighting=False,
                               q_weight=0.0, wdl_head=False)
        with h5.File(f"{folder}/Self_Play_Data.h5") as f:
            for i in range(0, 6, 2):
                z = f[f"values_{i}"][:, 0]
                if np.all(z == 0):
                    continue
                n = len(z)
                np.testing.assert_allclose(np.abs(z), 0.99 ** (n - 1 - np.arange(n)), rtol=1e-5)

    def test_plain_self_play_is_unchanged(self):
        folder, recorded = play_games(1, full_search_prob=1.0, cheap_policy_targets=False, surprise_weighting=False,
                                      value_discount=1.0)
        with h5.File(f"{folder}/Self_Play_Data.h5") as f:
            counts = f["search_stats"][:]
            self.assertEqual(counts[SEARCH_KINDS.index("cheap")], 0)
            self.assertEqual(recorded[0], counts[-2])
            self.assertEqual(counts[-1], 96 * counts[-2])


class EndgameSolverTest(unittest.TestCase):
    def test_races_are_played_by_the_solver(self):
        game = Quoridor()
        game.board[2, 0, :] = 0
        folder = tempfile.mkdtemp()
        Make_Dataset_File(folder)
        tc = qc.train_config.replace(**{**SETTINGS, "endgame_solver": True})
        session = FakeSession(game.get_input_state().shape, game.policy_shape[0], seed=0, scale=0.002)
        n = Self_Play(game, session, qc.build_config, tc, mp.Lock(), folder, 1).play()
        with h5.File(f"{folder}/Self_Play_Data.h5") as f:
            counts, stats = f["search_stats"][:], f["game_stats"][:]
            values, policies = f["values_0"][:], f["policies_0"][:]
        self.assertEqual(counts[SEARCH_KINDS.index("solver")], counts[-2])
        self.assertEqual((counts[-2], counts[-1]), (n, 0))
        winner = -1 if stats[3] else 1
        mover = np.where(np.arange(n) % 2 == 0, -1, 1)
        np.testing.assert_allclose(values[:, 0] - values[:, 2], mover * winner * 0.99 ** (n - 1 - np.arange(n)),
                                   rtol=1e-5)
        np.testing.assert_allclose(policies.sum(1), 1, atol=1e-5)

    def test_search_stats_from_before_the_solver_are_padded(self):
        folder = tempfile.mkdtemp()
        Make_Dataset_File(folder)
        with h5.File(f"{folder}/Self_Play_Data.h5", "r+") as f:
            f.create_dataset("search_stats", data=np.arange(len(SEARCH_KINDS) + 1, dtype=np.int64))
        game = Quoridor()
        session = FakeSession(game.get_input_state().shape, game.policy_shape[0], seed=0, scale=0.002)
        Self_Play(game, session, qc.build_config, qc.train_config.replace(**SETTINGS), mp.Lock(), folder,
                  1)._write(None, 0, 1, np.ones(len(SEARCH_KINDS) + 2, dtype=np.int64))
        with h5.File(f"{folder}/Self_Play_Data.h5") as f:
            counts = f["search_stats"][:]
        old = np.arange(len(SEARCH_KINDS) + 1)
        np.testing.assert_array_equal(counts, np.concatenate([old[:-2], [0], old[-2:]]) + 1)


class ExactSearchTest(unittest.TestCase):
    def test_exact_runs_the_requested_sims(self):
        game = Quoridor()
        session = FakeSession(game.get_input_state().shape, game.policy_shape[0], seed=0, scale=0.002)
        mcts = MCTS(game, session, use_dirichlet=False, tau=0.0, leaves_per_batch=4)
        mcts.run(iteration_limit=24, use_bar=False, exact=True)
        self.assertEqual(mcts.root.visits, 24)
        mcts.run(iteration_limit=72, use_bar=False, exact=True)
        self.assertEqual(mcts.root.visits, 96)

    def test_raw_priors_survive_root_noise(self):
        game = Quoridor()
        session = FakeSession(game.get_input_state().shape, game.policy_shape[0], seed=0, scale=0.002)
        np.random.seed(0)
        noisy = MCTS(game, session, use_dirichlet=True, dirichlet_alpha=0.08, tau=0.0)
        clean = MCTS(game, session, use_dirichlet=False, tau=0.0)
        order = {tuple(a): i for i, a in enumerate(clean.root.child_legal_actions)}
        idx = [order[tuple(a)] for a in noisy.root.child_legal_actions]
        np.testing.assert_allclose(noisy.root_priors(), clean.root_priors()[idx], rtol=1e-6)
        self.assertFalse(np.allclose(noisy.root.child_prob_priors, noisy.root_priors()))


if __name__ == "__main__":
    unittest.main()
