"""Quoridor alpha-beta solver against plain minimax. python -m unittest tests.test_solver"""
import unittest

import numpy as np

from Quoridor.game import Quoridor, SIZE, _legal_actions
from Quoridor.solver import MATE, Solver

N = SIZE


def minimax(board, player, depth, ply=0):
    """No pruning, no table: the reference."""
    opponent_row = board[2, 1 if -player == -1 else 2, 0]
    if opponent_row == (0 if -player == -1 else N - 1):
        return -(MATE - ply)
    if depth == 0:
        return 0
    best = -10 ** 9
    for t, r, c in _legal_actions(board, player):
        child = Quoridor.do_action_MCTS(board.copy(), np.array([t, r, c], dtype=np.uint8), player)
        best = max(best, -minimax(child, -player, depth - 1, ply + 1))
    return best


def random_position(rng, walls_each, walls_on_board, near):
    game = Quoridor()
    board = game.board
    placed = 0
    while placed < walls_on_board:
        legal = _legal_actions(board, -1)
        walls = legal[legal[:, 0] > 0]
        t, r, c = walls[rng.integers(len(walls))]
        board[t - 1, r, c] = 1
        placed += 1
    board[2, 0, 0] = board[2, 0, 1] = 0
    while True:
        black = (rng.integers(1, 1 + near), rng.integers(N))
        white = (N - 1 - rng.integers(1, 1 + near), rng.integers(N))
        if black != white:
            break
    board[2, 1, :2] = black
    board[2, 2, :2] = white
    board[2, 0, 0] = board[2, 0, 1] = walls_each
    return board, (1 if rng.random() < 0.5 else -1)


class SolverTest(unittest.TestCase):
    def check(self, seed, count, walls_each, walls_on_board, near, depth):
        rng = np.random.default_rng(seed)
        solver = Solver(tt_bits=16, max_depth=depth, node_budget=10 ** 8)
        proven = 0
        for _ in range(count):
            board, player = random_position(rng, walls_each, walls_on_board, near)
            reference = minimax(board, player, depth)
            result = solver.solve(board, player)
            if abs(reference) > MATE - 1000:
                self.assertIsNotNone(result)
                value, plies, action, _ = result
                self.assertEqual(value * (MATE - plies), reference)
                after = Quoridor.do_action_MCTS(board.copy(), action, player)
                if plies > 1:
                    follow = solver.solve(after, -player)
                    self.assertEqual((follow[0], follow[1]), (-value, plies - 1))
                proven += 1
            else:
                self.assertIsNone(result)
        return proven

    def test_races(self):
        self.assertGreater(self.check(0, 25, 0, 12, 2, 6), 10)

    def test_one_wall(self):
        self.assertGreater(self.check(1, 6, 1, 10, 1, 4), 2)

    def test_cpp_matches_numba(self):
        from pathlib import Path
        from Quoridor.solver import CppSolver
        if not (Path(__file__).resolve().parents[1] / "Quoridor" / "cpp" / "libsolver.so").exists():
            self.skipTest("build Quoridor/cpp/libsolver.so first")
        rng = np.random.default_rng(7)
        numba_solver = Solver(tt_bits=14, max_depth=12, node_budget=50_000)
        cpp_solver = CppSolver(tt_bits=14, max_depth=12, node_budget=50_000)
        for walls_each, near in ((0, 4), (1, 3), (2, 2)):
            for _ in range(15):
                board, player = random_position(rng, walls_each, 10, near)
                self.assertEqual(numba_solver.raw(board, player), cpp_solver.raw(board, player))

    def test_race_targets_keep_every_optimal_move(self):
        from Quoridor.solver import race_targets
        rng = np.random.default_rng(3)
        checked = 0
        for _ in range(20):
            board, player = random_position(rng, 0, 12, 2)
            reference = minimax(board, player, 6)
            if abs(reference) < MATE - 1000:
                continue
            optimal = {tuple(int(x) for x in a) for a in _legal_actions(board, player)
                       if -minimax(Quoridor.do_action_MCTS(board.copy(), a, player), -player, 5, 1) == reference}
            value, plies, keep = race_targets(board, player)
            self.assertEqual(value * (MATE - plies), reference)
            self.assertEqual({tuple(int(x) for x in a) for a in keep}, optimal)
            checked += 1
        self.assertGreater(checked, 5)

    def test_race_targets_need_no_walls(self):
        from Quoridor.solver import race_targets
        board = Quoridor().board.copy()
        board[2, 0, 1] = 0
        self.assertIsNone(race_targets(board, -1))

    def test_unknown_when_too_deep(self):
        board = Quoridor().board.copy()
        board[2, 0, 0] = board[2, 0, 1] = 0
        self.assertIsNone(Solver(tt_bits=12, max_depth=6).solve(board, -1))


if __name__ == "__main__":
    unittest.main()
