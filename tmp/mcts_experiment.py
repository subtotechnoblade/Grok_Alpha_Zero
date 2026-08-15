"""Controlled comparison: MCTS_original.py (pre-optimization) vs MCTS.py.

Policy and value are RANDOM throughout (session=None -> _get_dummy_outputs), so the
network is removed as a variable and only the search machinery is measured.

Design
  A. correctness    identical-seed game replay, forced-win detection, root policy
                    agreement over many positions
  B. throughput     fixed workload (same positions, same sim count), variants
                    INTERLEAVED across repeats so thermal/frequency drift cannot
                    favour whichever ran first; paired per-position differences
  C. realistic      full self-play games at the training config

Usage (WSL): python tmp/mcts_experiment.py [repeats]
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from Connect4.Connect4 import Connect4, train_config
sys.path.insert(0, os.path.join(ROOT, "tmp"))
import MCTS_original as base_mod   # pristine pre-optimization copy
import MCTS as fixed_mod            # the live, optimized MCTS.py

VARIANTS = [("MCTS_original", base_mod.MCTS), ("MCTS.py (optimized)", fixed_mod.MCTS)]
SIMS = int(train_config["MCTS_iteration_limit"] * 1.5)   # 375, what self play uses
REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 7

PUCT = train_config["c_puct_init"]
ALPHA = train_config["dirichlet_alpha"]


def make(cls, game, tau=1.0, dirichlet=True):
    # session=None -> random policy, 0.0 value. no network anywhere in this test.
    return cls(game, None, use_njit=True, c_puct_init=PUCT, use_dirichlet=dirichlet,
               dirichlet_alpha=ALPHA, tau=tau, fast_find_win=False)


def position(actions):
    g = Connect4()
    for a in actions:
        if int(np.sum(np.abs(g.board[:, a]))) < 6:
            g.do_action(a)
    return g


# a deterministic spread of positions: openings, midgame, late
rng = np.random.default_rng(20260726)
POSITIONS = []
for i in range(12):
    n = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24][i]
    acts = []
    g = Connect4()
    for _ in range(n):
        legal = [c for c in range(7) if int(np.sum(np.abs(g.board[:, c]))) < 6]
        if not legal or g.action_history and g.check_win() != -2:
            break
        a = int(rng.choice(legal))
        g.do_action(a)
        acts.append(a)
    POSITIONS.append(acts)


def play_game(cls, seed, sims=SIMS):
    np.random.seed(seed)
    g = Connect4()
    tree = make(cls, g)
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
    return moves, list(g.action_history), winner


print("=" * 78)
print(f"MCTS comparison | connect4 | random policy+value (session=None)")
print(f"sims/move {SIMS} | c_puct {PUCT} | dirichlet_alpha {ALPHA} | repeats {REPEATS}")

# warm every numba signature for BOTH modules before any timing
for _, cls in VARIANTS:
    g = Connect4()
    t = make(cls, g)
    t.run(50, use_bar=False)

# ------------------------------------------------------------------ A. correctness
print("=" * 78)
print("A. CORRECTNESS")

print("  A1 forced win at ply 6 (col 3 wins immediately)")
for name, cls in VARIANTS:
    g = position([0, 6, 1, 6, 2, 6])
    tree = cls(g, None, use_njit=True, c_puct_init=PUCT, use_dirichlet=False,
               tau=0.0, fast_find_win=True)
    a, _ = tree.run(300, use_bar=False)
    print(f"     {name:>14}: plays {int(a)}  {'OK' if int(a) == 3 else 'WRONG'}")

print("  A2 forced block (opponent threatens col 3)")
for name, cls in VARIANTS:
    g = position([0, 6, 1, 5, 2])          # -1 has 0,1,2 -> +1 must block at 3
    tree = cls(g, None, use_njit=True, c_puct_init=PUCT, use_dirichlet=False,
               tau=0.0, fast_find_win=True)
    a, _ = tree.run(400, use_bar=False)
    print(f"     {name:>14}: plays {int(a)}  {'OK' if int(a) == 3 else 'MISSED BLOCK'}")

print("  A3 root policy agreement over 12 positions, 600 sims, no noise")
kls, tops = [], []
for acts in POSITIONS:
    dists = []
    for _, cls in VARIANTS:
        np.random.seed(4242)
        g = position(acts)
        if g.action_history and g.check_win() != -2:
            dists = []
            break
        tree = cls(g, None, use_njit=True, c_puct_init=PUCT, use_dirichlet=False,
                   tau=0.0, fast_find_win=False)
        _, stats = tree.run(600, use_bar=False)
        d = np.zeros(7)
        for row in stats:
            d[int(row[0])] = float(row[1])
        dists.append(d)
    if len(dists) != 2:
        continue
    p, q = dists[0] + 1e-12, dists[1] + 1e-12
    kls.append(float(np.sum(p * np.log(p / q))))
    tops.append(int(np.argmax(dists[0]) == np.argmax(dists[1])))
print(f"     top-1 move agreement : {sum(tops)}/{len(tops)}")
print(f"     mean KL(base||fixed) : {np.mean(kls):.5f}  (max {np.max(kls):.5f})")
print("     note: FIX 2 accumulates in float64 not float32, so exact equality is")
print("           not expected; these should be small and unbiased, not zero.")

# ------------------------------------------------------------------ B. throughput
print("=" * 78)
print(f"B. THROUGHPUT, fixed workload ({len(POSITIONS)} positions x {SIMS} sims, "
      f"{REPEATS} interleaved repeats)")

timings = {name: {i: [] for i in range(len(POSITIONS))} for name, _ in VARIANTS}
nodes = {name: {} for name, _ in VARIANTS}
for r in range(REPEATS):
    for i, acts in enumerate(POSITIONS):
        for name, cls in VARIANTS:                       # interleaved, not batched
            np.random.seed(1000 + i)
            g = position(acts)
            if g.action_history and g.check_win() != -2:
                continue
            tree = make(cls, g, tau=0.0)
            t0 = time.perf_counter()
            tree.run(SIMS, use_bar=False)
            dt = time.perf_counter() - t0
            timings[name][i].append(dt / SIMS * 1e6)
            nodes[name][i] = int(sum(tree.root.child_visits))

print(f"  {'position':>10} {'ply':>4} {'base us/sim':>13} {'fixed us/sim':>13} "
      f"{'speedup':>9} {'root visits':>12}")
ratios = []
for i, acts in enumerate(POSITIONS):
    b = timings["MCTS_original"][i]
    f = timings["MCTS.py (optimized)"][i]
    if not b or not f:
        continue
    bm, fm = float(np.median(b)), float(np.median(f))
    ratios.append(bm / fm)
    same = nodes["MCTS_original"].get(i) == nodes["MCTS.py (optimized)"].get(i)
    print(f"  {i:>10} {len(acts):>4} {bm:>13.2f} {fm:>13.2f} {bm / fm:>8.2f}x "
          f"{nodes['MCTS_original'].get(i, 0):>7}{'  =' if same else ' ~='}")

allb = [v for i in timings["MCTS_original"] for v in timings["MCTS_original"][i]]
allf = [v for i in timings["MCTS.py (optimized)"] for v in timings["MCTS.py (optimized)"][i]]
print(f"  {'-' * 70}")
print(f"  base  : {np.mean(allb):7.2f} +/- {np.std(allb):5.2f} us/sim  "
      f"({1e6 / np.mean(allb):9,.0f} sims/s)")
print(f"  fixed : {np.mean(allf):7.2f} +/- {np.std(allf):5.2f} us/sim  "
      f"({1e6 / np.mean(allf):9,.0f} sims/s)")
print(f"  paired speedup per position: mean {np.mean(ratios):.2f}x, "
      f"min {np.min(ratios):.2f}x, max {np.max(ratios):.2f}x")

# ------------------------------------------------------------------ C. realistic
print("=" * 78)
print(f"C. FULL SELF-PLAY GAMES (tau schedule {train_config['num_explore_actions_first']}"
      f"/{train_config['num_explore_actions_second']}, dirichlet on)")
SEEDS = [11, 22, 33, 44, 55]
res = {}
for name, cls in VARIANTS:
    tot_moves, tot_wall = 0, 0.0
    for s in SEEDS:
        t0 = time.perf_counter()
        moves, _, _ = play_game(cls, s)
        tot_wall += time.perf_counter() - t0
        tot_moves += moves
    res[name] = (tot_moves, tot_wall)
    print(f"  {name:>14}: {len(SEEDS)} games, {tot_moves} moves, {tot_wall:6.2f}s, "
          f"{tot_moves * SIMS / tot_wall:9,.0f} sims/s, "
          f"{tot_wall / len(SEEDS):5.2f}s per game")
b_rate = res["MCTS_original"][0] * SIMS / res["MCTS_original"][1]
f_rate = res["MCTS.py (optimized)"][0] * SIMS / res["MCTS.py (optimized)"][1]
print(f"  speedup: {f_rate / b_rate:.2f}x")

print("\n  NOT extrapolated to real generation time on purpose. These runs have NO")
print("  network: session=None costs one np.random.uniform per leaf, where real")
print("  self play pays a shared-memory round trip to the inference server. Real")
print("  cost per sim is (MCTS + inference); only the MCTS term shrinks here, so")
print("  the end-to-end generation speedup is strictly less than the number above,")
print("  and inference becomes a larger share of the critical path as MCTS gets")
print("  cheaper. Measure a real generation before and after to get that number.")
