"""Per-first-move outcome and ply-0 target sharpness for one generation."""
import sys
import numpy as np
import h5py as h5
from collections import defaultdict

gen = sys.argv[1] if len(sys.argv) > 1 else "9"
path = f"Connect4/Grok_Zero_Train/{gen}/Self_Play_Data.h5"

with h5.File(path, "r") as f:
    n_datasets = (len(f.keys()) - 1) // 3
    outcome = defaultdict(list)   # first move -> z for player -1 (from value target at ply 0)
    peak = []
    for k in range(0, n_datasets, 2):
        b = f[f"boards_{k}"][:]
        if b.shape[0] < 2:
            continue
        diff = b[1][:, :, 3].astype(np.int16) - b[0][:, :, 3].astype(np.int16)
        xs = np.nonzero(diff)[1]
        first = int(xs[0])
        # value target at ply 0 = 0.8*z + 0.2*q from the mover's (player -1) perspective
        v0 = float(f[f"values_{k}"][0])
        outcome[first].append(v0)
        p0 = f[f"policies_{k}"][0]
        peak.append(float(np.max(p0)))

print(f"gen {gen}: ply-0 value target (0.8z+0.2q, player -1 perspective) by first move")
for m in sorted(outcome):
    v = np.array(outcome[m])
    print(f"  col {m}: n={len(v):4d}  mean={v.mean():+.3f}  (z-part sign: >0 favors p-1)")
print(f"ply-0 policy target peakiness: mean max prob = {np.mean(peak):.3f}, "
      f"median = {np.median(peak):.3f}")
