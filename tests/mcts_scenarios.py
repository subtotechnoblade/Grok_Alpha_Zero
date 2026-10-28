import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Connect4.game import Connect4
from Gomoku.game import Gomoku
from Session_Cache import Cache_Wrapper
from tests.fake_net import FakeSession

FORCED_BLOCK = [(7, 7), (6, 7), (8, 7), (0, 0), (9, 7), (0, 1), (10, 7)]  # white must block (11, 7)
IMMEDIATE_WIN = FORCED_BLOCK + [(0, 2)]  # black wins at (11, 7)

# name: game, session kind, opening, moves searched, sims, tau, fpu, tree reuse mode, seed
SCENARIOS = {
    "c4_gen0": (Connect4, None, [], 6, 200, 1.0, 0.0, "reuse", 1),
    "c4_fake_tau0": (Connect4, "fake", [3, 3], 6, 300, 0.0, 0.0, "reuse", 2),
    "c4_fake_tau1": (Connect4, "fake", [], 6, 300, 1.0, 0.0, "reuse", 3),
    "c4_fake_newroot": (Connect4, "fake", [2], 5, 250, 1.0, 0.0, "new_root", 4),
    "c4_cache": (Connect4, "cache", [], 6, 300, 1.0, 0.0, "reuse", 5),
    "gomoku_gen0": (Gomoku, None, [], 3, 200, 1.0, 0.2, "reuse", 6),
    "gomoku_fake_tau1": (Gomoku, "fake", [(7, 7)], 4, 400, 1.0, 0.2, "reuse", 7),
    "gomoku_fake_tau0": (Gomoku, "fake", [(7, 7), (8, 8)], 4, 400, 0.0, 0.2, "reuse", 8),
    "gomoku_newroot": (Gomoku, "fake", [(7, 7)], 3, 300, 1.0, 0.2, "new_root", 9),
    "gomoku_cache": (Gomoku, "cache", [], 3, 300, 1.0, 0.2, "reuse", 10),
    "gomoku_block": (Gomoku, "fake", FORCED_BLOCK, 2, 300, 0.0, 0.2, "reuse", 11),
    "gomoku_win": (Gomoku, "fake", IMMEDIATE_WIN, 1, 300, 0.0, 0.2, "reuse", 12),
}


def make_session(kind, game):
    if kind is None:
        return None
    fake = FakeSession(game.get_input_state().shape, game.policy_shape[0], seed=0)
    return fake if kind == "fake" else Cache_Wrapper(fake, max_cache_depth=4)


def snapshot(mcts, move, move_probs):
    root = mcts.root
    return {
        "move": np.array(move).copy(),
        "root_visits": int(root.visits),
        "child_visits": np.array(root.child_visits).copy(),
        "child_values": np.array(root.child_values).copy(),
        "child_prob_priors": np.array(root.child_prob_priors).copy(),
        "move_probs": [[np.array(entry[0]).copy(), *[float(v) if v is not None else None for v in entry[1:]]]
                       for entry in move_probs],
    }


def run_scenario(mcts_class, name, **mcts_kwargs):
    game_class, kind, opening, n_moves, sims, tau, fpu, mode, seed = SCENARIOS[name]
    np.random.seed(seed)
    game = game_class()
    for action in opening:
        game.do_action(action)
    session = make_session(kind, game)
    mcts = mcts_class(game, session, c_puct_init=2.0, fpu_reduction=fpu, use_dirichlet=True,
                      dirichlet_alpha=0.3, tau=tau, **mcts_kwargs)
    records = []
    for _ in range(n_moves):
        if game.action_history and game.check_win() != -2:
            break
        move, move_probs = mcts.run(iteration_limit=sims, use_bar=False)
        records.append(snapshot(mcts, move, move_probs))
        game.do_action(move)
        mcts.prune_tree(move, create_new_root=(mode == "new_root"))
    stats = session.stats() if isinstance(session, Cache_Wrapper) else None
    return {"records": records, "cache_stats": stats}
