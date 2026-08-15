"""Per-phase before/after profile: MCTS_original.py (pre-optimization) vs MCTS.py.

Random policy and value (session=None), so no network is involved. Because the
dummy value is exactly 0.0 - bit identical in float32 and float64 - both variants
play the SAME games from the same seed, which makes this a true paired comparison:
identical call counts, identical work, only the implementation differs.

Phases are nested. _expand calls _compute_outputs and _expand_with_terminal_actions,
so "expand exclusive" below subtracts them out.

Usage (WSL): python tmp/mcts_phase_profile.py [n_seeds]
"""
import os
import sys
import time
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from Connect4.Connect4 import Connect4, train_config
sys.path.insert(0, os.path.join(ROOT, "tmp"))
import MCTS_original as base_mod   # pristine pre-optimization copy
import MCTS as fixed_mod            # the live, optimized MCTS.py

SIMS = int(train_config["MCTS_iteration_limit"] * 1.5)
SEEDS = list(range(101, 101 + (int(sys.argv[1]) if len(sys.argv) > 1 else 4)))

PHASES = [("_PUCT_select", "select"),
          ("_expand", "expand"),
          ("_expand_with_terminal_actions", "terminal"),
          ("_compute_outputs", "inference"),
          ("_back_propagate", "backprop")]


def play_game(cls, seed, sims=SIMS):
    np.random.seed(seed)
    g = Connect4()
    tree = cls(g, None, use_njit=True, c_puct_init=train_config["c_puct_init"],
               use_dirichlet=True, dirichlet_alpha=train_config["dirichlet_alpha"],
               tau=1.0, fast_find_win=False)
    winner, moves = -2, 0
    while winner == -2 and moves < train_config["max_actions"]:
        if moves % 2 == 0:
            explore = moves // 2 < train_config["num_explore_actions_first"]
        else:
            explore = (moves + 1) // 2 < train_config["num_explore_actions_second"]
        tree.update_hyperparams(tau=1.0 if explore else 0.0)
        action, _ = tree.run(sims, use_bar=False)
        g.do_action(action)
        winner = g.check_win()
        if winner == -2:
            tree.prune_tree(action)
        moves += 1
    return moves, list(g.action_history)


def run_plain(cls):
    t0 = time.perf_counter()
    out = [play_game(cls, s) for s in SEEDS]
    return time.perf_counter() - t0, out


def run_instrumented(cls):
    acc, cnt = Counter(), Counter()
    saved = {}
    for meth, key in PHASES:
        orig = getattr(cls, meth)
        saved[meth] = orig

        def make(o, k):
            def w(*a, **kw):
                t0 = time.perf_counter_ns()
                try:
                    return o(*a, **kw)
                finally:
                    acc[k] += time.perf_counter_ns() - t0
                    cnt[k] += 1
            return w
        setattr(cls, meth, make(orig, key))
    t0 = time.perf_counter()
    out = [play_game(cls, s) for s in SEEDS]
    wall = time.perf_counter() - t0
    for meth, orig in saved.items():
        setattr(cls, meth, orig)
    return acc, cnt, wall, out


# warm numba for both modules
for cls in (base_mod.MCTS, fixed_mod.MCTS):
    t = cls(Connect4(), None, use_njit=True, tau=0.0)
    t.run(50, use_bar=False)

print("=" * 80)
print(f"MCTS phase profile | connect4 | random policy+value | {SIMS} sims/move | "
      f"{len(SEEDS)} games")

plain_b, games_b = run_plain(base_mod.MCTS)
plain_f, games_f = run_plain(fixed_mod.MCTS)
assert [g[1] for g in games_b] == [g[1] for g in games_f], \
    "variants played different games - not a paired comparison"
moves = sum(g[0] for g in games_b)
sims = moves * SIMS
print(f"identical games confirmed: {moves} moves, ~{sims:,} sims per variant")
print(f"uninstrumented wall: base {plain_b:.2f}s, fixed {plain_f:.2f}s  "
      f"-> {plain_b / plain_f:.2f}x")

acc_b, cnt_b, wall_b, _ = run_instrumented(base_mod.MCTS)
acc_f, cnt_f, wall_f, _ = run_instrumented(fixed_mod.MCTS)
print(f"instrumented wall  : base {wall_b:.2f}s, fixed {wall_f:.2f}s  "
      f"(timer overhead inflates both ~{wall_b / plain_b:.2f}x / "
      f"{wall_f / plain_f:.2f}x)")

# exclusive expand = expand - the two things it calls
for acc, cnt in ((acc_b, cnt_b), (acc_f, cnt_f)):
    acc["expand_excl"] = acc["expand"] - acc["inference"] - acc["terminal"]
    cnt["expand_excl"] = cnt["expand"]

print("=" * 80)
print(f"{'phase':>18} {'calls':>9} {'base us':>9} {'fixed us':>9} {'speedup':>8} "
      f"{'base s':>8} {'fixed s':>8} {'saved s':>8}")
ROWS = [("select", "select"),
        ("expand (inclusive)", "expand"),
        ("  expand exclusive", "expand_excl"),
        ("  terminal expand", "terminal"),
        ("  inference (dummy)", "inference"),
        ("backprop", "backprop")]
for label, key in ROWS:
    if not cnt_b.get(key):
        continue
    ub = acc_b[key] / cnt_b[key] / 1000
    uf = acc_f[key] / cnt_f[key] / 1000
    sb, sf = acc_b[key] / 1e9, acc_f[key] / 1e9
    print(f"{label:>18} {cnt_b[key]:>9,} {ub:>9.2f} {uf:>9.2f} {ub / uf:>7.2f}x "
          f"{sb:>8.3f} {sf:>8.3f} {sb - sf:>8.3f}")

print("=" * 80)
print("share of instrumented wall time")
print(f"{'phase':>18} {'base %':>9} {'fixed %':>9}")
for label, key in ROWS:
    if not cnt_b.get(key) or key == "expand":
        continue
    print(f"{label:>18} {acc_b[key] / 1e9 / wall_b * 100:>8.1f}% "
          f"{acc_f[key] / 1e9 / wall_f * 100:>8.1f}%")

print("=" * 80)
print("per simulation (us), the number that matters")
tot_b = tot_f = 0.0
for label, key in ROWS:
    if not cnt_b.get(key) or key == "expand":
        continue
    pb = acc_b[key] / 1e3 / sims
    pf = acc_f[key] / 1e3 / sims
    tot_b += pb
    tot_f += pf
    print(f"{label:>18} {pb:>9.2f} {pf:>9.2f}   delta {pf - pb:+8.2f}")
print(f"{'accounted total':>18} {tot_b:>9.2f} {tot_f:>9.2f}   delta {tot_f - tot_b:+8.2f}")
print(f"{'measured us/sim':>18} {plain_b / sims * 1e6:>9.2f} {plain_f / sims * 1e6:>9.2f}"
      f"   (uninstrumented)")
