"""Is the opener finally CONVERTING the center opening? Per gen:
mean value target at ply 0 (opener's perspective) conditioned on opening col."""
import os
import sys
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import h5py as h5

GENS = sorted(glob.glob("Connect4/Grok_Zero_Train/*"),
              key=lambda p: int(os.path.basename(p)))

print("mean ply-0 value target (opener's view) by opening column")
print("z > 0: opener wins.  connect4 theory: col 3 should converge to +1")
print(f"{'gen':>4} {'col3 n':>7} {'col3 mean':>10} {'off-center n':>13} {'off-center mean':>16}")
for d in GENS:
    f = os.path.join(d, "Self_Play_Data.h5")
    if not os.path.exists(f):
        continue
    try:
        with h5.File(f, "r") as h:
            n = int(h["game_stats"][2])
            n_sets = (len(h.keys()) - 1) // 3
            if n <= 0 or n_sets % n:
                continue
            aug = n_sets // n
            c3, off = [], []
            for gid in range(n):
                did = gid * aug
                brd = h[f"boards_{did}"]
                if brd.shape[0] < 2:
                    continue
                b0, b1 = brd[0][..., -1], brd[1][..., -1]
                diff = np.argwhere(b0 != b1)
                if len(diff) == 0:
                    continue
                col = int(diff[0][1])
                z0 = float(h[f"values_{did}"][0][0])
                (c3 if col == 3 else off).append(z0)
    except OSError:
        print(f"{os.path.basename(d):>4}  (h5 busy, skipped)")
        continue
    if c3 or off:
        cm = np.mean(c3) if c3 else float("nan")
        om = np.mean(off) if off else float("nan")
        print(f"{os.path.basename(d):>4} {len(c3):>7} {cm:>+10.3f} {len(off):>13} {om:>+16.3f}")
