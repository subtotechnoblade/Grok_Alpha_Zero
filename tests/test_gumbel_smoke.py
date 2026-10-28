"""R14: Gumbel MCTS still runs, including through the reworked Cache_Wrapper.
python -m unittest tests.test_gumbel_smoke"""
import unittest

import numpy as np

from Connect4.game import Connect4
from Gomoku.game import Gomoku
from MCTS_Gumbel import MCTS_Gumbel
from Session_Cache import Cache_Wrapper
from tests.fake_net import FakeSession


class TestGumbelSmoke(unittest.TestCase):
    def test_games(self):
        for game_class in (Connect4, Gomoku):
            for cached in (False, True):
                with self.subTest(game=game_class.__name__, cached=cached):
                    np.random.seed(0)
                    game = game_class()
                    session = FakeSession(game.get_input_state().shape, game.policy_shape[0])
                    if cached:
                        session = Cache_Wrapper(session, 4)
                    mcts = MCTS_Gumbel(game, session, use_gumbel_noise=True, m=8)
                    for _ in range(3):
                        move, probs = mcts.run(iteration_limit=64, use_bar=False)
                        self.assertTrue(len(probs) > 0)
                        game.do_action(move)
                        mcts.prune_tree(move)
                    if cached:
                        self.assertGreater(session.stats()["misses"], 0)


if __name__ == "__main__":
    unittest.main()
