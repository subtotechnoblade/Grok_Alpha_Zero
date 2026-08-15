"""M2: per-gen self-play data stats at the root, READ-ONLY.

Per gen: mean ply-0 policy target (root visit dist), games per opening column,
mean ply-0 value target per opening column (opener's perspective), and
-mean ply-1 value target per opening column (this trains the child value the
root PUCT actually consumes; sign flipped to opener's perspective).
"""
import os, sys
ROOT = r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import h5py as h5

GENS = list(range(3, 13))
out = {}

for gen in GENS:
    fp = f"Connect4/Grok_Zero_Train/{gen}/Self_Play_Data.h5"
    if not os.path.exists(fp):
        continue
    try:
        with h5.File(fp, "r") as f:
            stats = f["game_stats"][:]
            n = int(stats[2])
            n_sets = (len(f.keys()) - 1) // 3
            if n <= 0 or n_sets % n:
                print(f"gen {gen}: inconsistent ({n} games, {n_sets} sets) - mid-write, skipped")
                continue
            aug = n_sets // n
            pol0 = np.zeros(7)
            cnt = np.zeros(7)          # games per opening col
            v0 = [[] for _ in range(7)]   # ply-0 value target (opener persp)
            v1 = [[] for _ in range(7)]   # ply-1 value target (p+1 persp)
            for gid in range(n):
                did = gid * aug
                p = f[f"policies_{did}"][0]
                pol0 += p
                b1 = f[f"boards_{did}"][1][:, :, 3]
                loc = np.argwhere(b1 != 0)
                assert len(loc) == 1
                c = int(loc[0][1])
                cnt[c] += 1
                vals = f[f"values_{did}"][:]
                v0[c].append(float(vals[0][0]))
                v1[c].append(float(vals[1][0]))
            wins = (int(stats[3]), int(stats[4]), int(stats[5]))
    except OSError as e:
        print(f"gen {gen}: h5 busy ({e}), skipped")
        continue

    pol0 /= n
    mv0 = np.array([np.mean(v0[c]) if v0[c] else np.nan for c in range(7)])
    mv1 = np.array([-np.mean(v1[c]) if v1[c] else np.nan for c in range(7)])
    out[gen] = dict(pol0=pol0, cnt=cnt, mv0=mv0, mv1=mv1, n=n, wins=wins)

    print(f"\n=== gen {gen}  ({n} games; p-1/draw/p+1 = {wins}) ===")
    print("            " + "".join(f"  col{c}  " for c in range(7)))
    print("root visits " + "".join(f" {pol0[c]:6.3f} " for c in range(7)))
    print("open count  " + "".join(f" {int(cnt[c]):6d} " for c in range(7)))
    print("v0|c (opnr) " + "".join(f" {mv0[c]:6.3f} " for c in range(7)))
    print("-v1|c(opnr) " + "".join(f" {mv1[c]:6.3f} " for c in range(7)))

np.savez(r"tmp\fresh2_cycle\data_stats.npz",
         **{f"g{g}_{k}": v for g, d in out.items() for k, v in d.items() if k != "wins"})
