"""Why does player +1 win in the new run? Read-only over the stored h5.

Three questions:
  1. how many games end on the LAST ply (42)? that ply is always p+1's, so the
     move-42 fix converts those from draws into p+1 wins - is that the shift?
  2. win rate conditioned on p-1's OPENING column. connect4 is a first-player
     win ONLY with a centre opening; tau=1 randomises p-1's first move, so if
     p-1 loses whenever it opens off-centre, exploration is destroying the
     first-player advantage rather than any bug doing it.
  3. value-target sign audit on the NEW data (the old audit cleared the old run).
"""
import os
import sys
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import h5py as h5

from Connect4.game import Connect4

GENS = sorted(glob.glob("Connect4/Grok_Zero_Train/*"),
              key=lambda p: int(os.path.basename(p)))


def load(gen_dir):
    """Yield (first_action, plies, winner_from_z, final_value_target) per game."""
    f = os.path.join(gen_dir, "Self_Play_Data.h5")
    with h5.File(f, "r") as h:
        n_games = int(h["game_stats"][2])
        n_sets = (len(h.keys()) - 1) // 3
        if n_games <= 0 or n_sets % n_games:
            return
        aug = n_sets // n_games
        for gid in range(n_games):
            did = gid * aug                       # un-mirrored copy only
            pol = h[f"policies_{did}"][:]
            val = h[f"values_{did}"][:]
            brd = h[f"boards_{did}"][:]
            yield brd, pol, val


print("1. GAMES ENDING ON THE FINAL PLY (always player +1's move)")
print(f"{'gen':>4} {'games':>6} {'len42':>6} {'pct':>7}")
for d in GENS:
    n = tot = 0
    for brd, pol, val in load(d):
        tot += 1
        if len(brd) >= 42:
            n += 1
    if tot:
        print(f"{os.path.basename(d):>4} {tot:>6} {n:>6} {n / tot:>6.1%}")

print()
print("2. OUTCOME BY PLAYER -1'S OPENING COLUMN  (z of the FIRST ply, mover = p-1)")
print("   z>0 means the opener won, z<0 means it lost")
for d in GENS[-3:]:
    counts = {}
    for brd, pol, val in load(d):
        # first ply's board is empty -> the opening column is argmax of the move
        # actually recover it from ply-1 board: the disc that appeared
        if len(brd) < 2:
            continue
        # plane 0 is a constant side-to-move plane (flips globally every ply, so
        # comparing all planes finds column 0 every time). Plane -1 is the board.
        b0, b1 = brd[0][..., -1], brd[1][..., -1]
        diff = np.argwhere(b0 != b1)
        if len(diff) == 0:
            continue
        col = int(diff[0][1])
        z0 = float(val[0][0])          # value target at ply 0, p-1 perspective
        counts.setdefault(col, []).append(z0)
    print(f"  gen {os.path.basename(d)}:")
    for col in sorted(counts):
        v = np.array(counts[col])
        print(f"    col {col}: {len(v):>4} games, mean target {v.mean():+.3f}")

print()
print("3. VALUE TARGET SIGN CHECK, split by decided vs drawn (latest gen)")
d = GENS[-1]
bad_decided = checked_decided = drawn = 0
for brd, pol, val in load(d):
    v = val[:, 0]
    # |0.8z| = 0.8 dominates |0.2q| <= 0.2, so a DECIDED game must alternate.
    # a draw has z = 0 everywhere, leaving target = 0.2q, |v| <= 0.2 - no
    # alternation required
    if np.abs(v).max() <= 0.25:
        drawn += 1
        continue
    checked_decided += 1
    s = np.sign(v)
    nz = s[s != 0]
    if len(nz) >= 4 and not np.all(nz[:-1] * nz[1:] <= 0):
        bad_decided += 1
print(f"  decided games: {checked_decided}, non-alternating: {bad_decided}")
print(f"  drawn games (|target| <= 0.2 throughout): {drawn}")

print()
print("4. WHO WINS THE FULL-LENGTH (42 ply) GAMES, latest 3 gens")
for d in GENS[-3:]:
    p1 = pm1 = dr = 0
    for brd, pol, val in load(d):
        if len(brd) < 42:
            continue
        last = float(val[-1][0])       # last ply's mover is p+1
        if last > 0.25:
            p1 += 1
        elif last < -0.25:
            pm1 += 1
        else:
            dr += 1
    tot = p1 + pm1 + dr
    if tot:
        print(f"  gen {os.path.basename(d)}: {tot:>3} full games -> "
              f"p+1 wins {p1:>3} ({p1/tot:.0%}), p-1 wins {pm1:>3}, draws {dr:>3}")
