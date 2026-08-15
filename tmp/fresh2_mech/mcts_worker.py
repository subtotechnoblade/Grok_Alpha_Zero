"""Gen-0 (session=None) self-play reproduction, faithful to Self_Play.play().

Matches production defaults from Connect4/config.py + Self_Play.py:
  session=None, use_njit=True, c_puct_init=1.5, dirichlet_alpha=1.33,
  dirichlet_epsilon=0.25, fast_find_win=False, iteration_limit=int(250*1.5)=375,
  tau schedule: 3 explore plies per player, create_new_root=True.

usage: mcts_worker.py <n_games> <sims> <dirichlet 0/1> <tau_mode sched|zero>
                      <ffw 0/1> <out_csv> <seed>
"""
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero")
sys.path.insert(0, str(ROOT))

import numpy as np
from Connect4.game import Connect4
from MCTS import MCTS

NUM_EXPLORE_FIRST = 3
NUM_EXPLORE_SECOND = 3


def play_one(sims, use_dirichlet, tau_mode, ffw):
    game = Connect4()
    mcts = MCTS(game=game,
                session=None,
                use_njit=True,
                c_puct_init=1.5,
                use_dirichlet=use_dirichlet,
                dirichlet_alpha=1.33,
                dirichlet_epsilon=0.25,
                tau=1.0,
                fast_find_win=ffw)

    winner = -2
    while winner == -2 and len(game.action_history) < 42:
        cur = len(game.action_history)
        if tau_mode == "sched":
            if cur % 2 == 0:
                explore = cur // 2 < NUM_EXPLORE_FIRST
            else:
                explore = (cur - 1) // 2 < NUM_EXPLORE_SECOND
        else:  # "zero"
            explore = False
        mcts.update_hyperparams(tau=1.0 if explore else 0)

        action, _ = mcts.run(iteration_limit=sims, time_limit=None, use_bar=False)
        game.do_action(action)
        winner = game.check_win()
        if winner == -2:
            mcts.prune_tree(action, create_new_root=True)  # production setting
    if winner == -2:
        winner = 0
    return int(winner), len(game.action_history)


def main():
    n_games = int(sys.argv[1])
    sims = int(sys.argv[2])
    use_dirichlet = bool(int(sys.argv[3]))
    tau_mode = sys.argv[4]
    ffw = bool(int(sys.argv[5]))
    out_csv = sys.argv[6]
    seed = int(sys.argv[7])
    np.random.seed(seed)  # MCTS uses the global np.random state

    t0 = time.perf_counter()
    with open(out_csv, "w") as f:
        f.write("winner,length\n")
        for g in range(n_games):
            w, l = play_one(sims, use_dirichlet, tau_mode, ffw)
            f.write(f"{w},{l}\n")
            f.flush()
    wall = time.perf_counter() - t0
    print(f"done n={n_games} sims={sims} dir={use_dirichlet} tau={tau_mode} "
          f"ffw={ffw} wall={wall:.1f}s ({wall / n_games:.2f}s/game)")


if __name__ == "__main__":
    main()
