"""Did self play stop exploring? First-move distribution, position reuse, policy entropy."""
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import h5py as h5

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = "Connect4/Grok_Zero_Train"


def first_move_col(b):
    """Column filled between ply 0 and ply 1, from the occupancy plane."""
    occ0 = b[0][..., -1] != 0
    occ1 = b[1][..., -1] != 0
    diff = np.argwhere(occ1 & ~occ0)
    return int(diff[0][1]) if len(diff) else -1


print(f"{'gen':>4} {'games':>6} {'uniq%':>6} {'H(root)':>8} {'openings':>9}   first-move column mix")
print("-" * 96)
for d in sorted(glob(f"{RUN}/*/"), key=lambda p: int(Path(p).name)):
    path = os.path.join(d, "Self_Play_Data.h5")
    if not os.path.exists(path):
        continue
    with h5.File(path, "r") as f:
        n = int(f["game_stats"][2])
        if n == 0:
            continue
        aug = ((len(f.keys()) - 1) // 3) // n
        cols, ents, boards = [], [], []
        for g in range(n):
            i = g * aug
            b = f[f"boards_{i}"][:]
            if len(b) < 2:
                continue
            cols.append(first_move_col(b))
            p0 = np.clip(f[f"policies_{i}"][0], 1e-12, 1)
            ents.append(float(-(p0 * np.log(p0)).sum()))
            boards.append(b)
    allb = np.concatenate(boards)
    uniq = len({x.tobytes() for x in allb}) / len(allb)
    hist = np.bincount(np.array(cols), minlength=7)
    mix = " ".join(f"{c}:{h/len(cols):>4.0%}" for c, h in enumerate(hist))
    nz = int((hist > 0).sum())
    print(f"{int(Path(d).name):>4} {n:>6} {uniq:>5.1%} {np.mean(ents):>8.4f} "
          f"{nz:>9}   {mix}")
