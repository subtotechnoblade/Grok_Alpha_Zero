"""CPU-only K=1 overhead of the batched loop vs the frozen pre-virtual-loss MCTS (gen 0, no network).
python -m tests.bench_k1_overhead"""
import os
import sys
import time

import numpy as np

from tests.mcts_scenarios import ROOT
from Gomoku.game import Gomoku
from Connect4.game import Connect4
from MCTS import MCTS as NewMCTS

sys.path.insert(0, os.path.join(ROOT, "tests", "data"))
from MCTS_before_vl import MCTS as OldMCTS


def timed(mcts_class, game_class, sims, moves, seed):
    np.random.seed(seed)
    game = game_class()
    mcts = mcts_class(game, None, c_puct_init=2.0, fpu_reduction=0.2, use_dirichlet=True, dirichlet_alpha=0.3, tau=1.0)
    total = 0
    start = time.process_time()
    for _ in range(moves):
        move, _ = mcts.run(iteration_limit=sims, use_bar=False)
        total += sims
        game.do_action(move)
        mcts.prune_tree(move)
    return total / (time.process_time() - start)


if __name__ == "__main__":
    for game_class, sims, moves in ((Gomoku, 2000, 6), (Connect4, 2000, 8)):
        timed(OldMCTS, game_class, 200, 2, 0)
        timed(NewMCTS, game_class, 200, 2, 0)
        old, new, ratios = [], [], []
        for rep in range(12):
            first, second = (OldMCTS, NewMCTS) if rep % 2 == 0 else (NewMCTS, OldMCTS)
            a = timed(first, game_class, sims, moves, rep)
            b = timed(second, game_class, sims, moves, rep)
            o, n = (a, b) if first is OldMCTS else (b, a)
            old.append(o)
            new.append(n)
            ratios.append(n / o)
        print(f"{game_class.__name__}: before {np.median(old):,.0f} sims/s  after {np.median(new):,.0f} sims/s  "
              f"paired ratio median {np.median(ratios):.3f} (IQR {np.percentile(ratios, 25):.3f}-"
              f"{np.percentile(ratios, 75):.3f})", flush=True)
