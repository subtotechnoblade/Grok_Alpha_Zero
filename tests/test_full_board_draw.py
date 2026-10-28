"""Gomoku full-board draws. python -m unittest tests.test_full_board_draw"""
import unittest

import numpy as np

from Gomoku.game import Gomoku
from MCTS import MCTS
from tests.mcts_scenarios import make_session


def draw_order():
    """225 alternating moves (first player -1) that fill the board without five in a row for either side."""
    owner = np.array([[-1 if ((x + 2 * y) // 2) % 2 == 0 else 1 for x in range(15)] for y in range(15)])
    first = [(x, y) for y in range(15) for x in range(15) if owner[y, x] == -1]
    second = [(x, y) for y in range(15) for x in range(15) if owner[y, x] == 1]
    assert len(first) == 113 and len(second) == 112
    moves = []
    for i in range(112):
        moves += [first[i], second[i]]
    return moves + [first[112]]


def play(moves):
    game = Gomoku()
    for move in moves:
        assert not game.action_history or game.check_win() == -2
        game.do_action(move)
    return game


class TestFullBoardDraw(unittest.TestCase):
    def test_filling_the_board_is_a_draw(self):
        game = play(draw_order())
        self.assertEqual(game.check_win(), 0)

    def test_no_win_before_the_last_move(self):
        moves = draw_order()
        game = Gomoku()
        for move in moves[:-1]:
            game.do_action(move)
            self.assertEqual(game.check_win(), -2)

    def test_search_reaches_the_full_board(self):
        moves = draw_order()
        for left, k in ((1, 1), (1, 8), (7, 1), (7, 8), (12, 8)):
            with self.subTest(empty_cells=left, leaves_per_batch=k), np.errstate(all="ignore"):
                game = play(moves[:-left])
                np.random.seed(0)
                mcts = MCTS(game, make_session("cached", game), c_puct_init=2.0, fpu_reduction=0.2,
                            use_dirichlet=False, tau=0.0, leaves_per_batch=k)
                while game.check_win() == -2:
                    move, _ = mcts.run(iteration_limit=600, use_bar=False)
                    game.do_action(move)
                    if game.check_win() == -2:
                        mcts.prune_tree(move)
                self.assertEqual(len(game.action_history), 225)
                self.assertEqual(game.check_win(), 0)


if __name__ == "__main__":
    unittest.main()
