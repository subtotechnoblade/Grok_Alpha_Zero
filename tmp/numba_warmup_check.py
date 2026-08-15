"""Was the use_njit comparison warmed up properly, and does numba recompile per MCTS?

MCTS.__init__ does:
    self.get_terminal_actions = nb.njit(self.get_terminal_actions_fn,
                                        cache=hasattr(os, "fork"))
On Windows hasattr(os,'fork') is False, so cache=False -> no on-disk cache. Every
MCTS instance builds a NEW dispatcher. Self play constructs one MCTS per game, so
if that recompiles each time it is a real per-game tax.

Run with the tensorflow_updated env.
"""
import os
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
from Connect4.Connect4 import Connect4, train_config
from MCTS import MCTS

SIMS = int(train_config["MCTS_iteration_limit"] * 1.5)
print(f"hasattr(os,'fork') = {hasattr(os,'fork')}  ->  numba cache={hasattr(os,'fork')}")
print()

# ------------------------------------------------------- 1. per-instance construction
print("1. MCTS construction cost (does the njit dispatcher recompile each time?)")
for use_njit in (True, False):
    ts = []
    for i in range(6):
        g = Connect4()
        t0 = time.perf_counter()
        tree = MCTS(g, None, use_njit=use_njit, tau=0.0, use_dirichlet=False)
        tree.run(30, use_bar=False)          # force first real call through the dispatcher
        ts.append(time.perf_counter() - t0)
    print(f"   use_njit={str(use_njit):<5} construct+30sims per instance (s): "
          + " ".join(f"{v:6.3f}" for v in ts))
    print(f"        first={ts[0]:.3f}s  median-of-rest={np.median(ts[1:]):.3f}s  "
          f"-> {'RECOMPILES each time' if np.median(ts[1:]) > ts[0] * 0.5 else 'compiled once, reused'}")
print()

# ------------------------------------------------------- 2. properly warmed comparison
print("2. use_njit True vs False, heavy warmup, interleaved, 5 reps")


def play(use_njit, n_moves=10, sims=SIMS, seed=7):
    np.random.seed(seed)
    g = Connect4()
    t = MCTS(g, None, use_njit=use_njit, c_puct_init=train_config["c_puct_init"],
             use_dirichlet=False, tau=0.0, fast_find_win=False)
    t0 = time.perf_counter()
    done = 0
    for _ in range(n_moves):
        a, _ = t.run(sims, use_bar=False)
        g.do_action(a)
        done += 1
        if g.check_win() != -2:
            break
        t.prune_tree(a)
    return time.perf_counter() - t0, done


# heavy warmup: full sim count, enough moves to reach terminal-expansion paths
for u in (True, False):
    for _ in range(3):
        play(u, n_moves=12, sims=SIMS, seed=1)

res = {True: [], False: []}
for rep in range(5):
    for u in (True, False):                      # interleaved
        dt, moves = play(u, seed=20 + rep)
        res[u].append(dt / moves)

for u in (True, False):
    a = np.array(res[u])
    print(f"   use_njit={str(u):<5} {np.median(a) * 1000:7.1f} ms/move  "
          f"(+/- {a.std() * 1000:.1f})  {SIMS / np.median(a):9,.0f} sims/s")
r = np.median(res[False]) / np.median(res[True])
print(f"   -> leaving use_njit=None (False on Windows) is {r:.2f}x "
      f"{'slower' if r > 1 else 'faster'} than forcing True")
