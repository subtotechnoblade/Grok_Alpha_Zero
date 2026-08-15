"""Mean ply-0 POLICY TARGET per gen = average root visit distribution at the
empty board. Shows what the search actually recommends, vs the raw net prior."""
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

print("average ROOT VISIT distribution at the empty board (ply-0 policy target)")
print(f"{'gen':>4}   col0  col1  col2  col3  col4  col5  col6")
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
            acc = np.zeros(7)
            cnt = 0
            for gid in range(n):
                did = gid * aug
                acc += h[f"policies_{did}"][0]
                cnt += 1
    except OSError:
        print(f"{os.path.basename(d):>4}  (h5 busy, skipped)")
        continue
    p = acc / cnt
    print(f"{os.path.basename(d):>4}   " + "  ".join(f"{x:.2f}" for x in p))
