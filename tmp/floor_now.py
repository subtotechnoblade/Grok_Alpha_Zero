"""Exact noise floor of the val value loss as keras computes it.

val value loss = sum_i w_i (pred_i - t_i)^2 / sum_i w_i over TEST-pool uniques,
w_i = n_i (mean-1). t_i = mean of the n_i test draws. A perfect predictor of the
position's true expectation mu_i leaves E[(t_i - mu_i)^2] = sigma_i^2 / n_i, so
floor = sum_i sigma_i^2 / S_test  (n_i cancels; once-only positions contribute
their FULL sigma^2 per sample - they dominate the test pool).

sigma_i^2 estimated from the position's occurrences across the WHOLE 8-11 window
(all games, ~10x the test draws) when n_window >= 3, else the per-ply-bucket
per-sample sigma^2 of repeated positions.

Also: empty-board drift significance (per-gen mean vs pooled, z-scores).
"""
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import h5py as h5

GENS = [15, 16, 17, 18]
TEST_PCT = {11: 0.10, 10: 0.09, 9: 0.08, 8: 0.07}
PLY_BUCKETS = [(0, 5), (6, 15), (16, 41)]


def main():
    rng = np.random.default_rng(1)
    window = defaultdict(list)          # key -> all window values (un-mirrored)
    ply_of = {}
    test_pool = defaultdict(list)       # replicated test split (mirrored, like training)
    for g in GENS:
        with h5.File(f"Connect4/Grok_Zero_Train/{g}/Self_Play_Data.h5", "r") as f:
            n = int(f["game_stats"][2])
            aug = ((len(f.keys()) - 1) // 3) // n
            test_ids = set(rng.permutation(n)[:int(TEST_PCT[g] * n)].tolist())
            for gid in range(n):
                b0 = f[f"boards_{gid * aug}"][:]
                v0 = f[f"values_{gid * aug}"][:]
                for ply, (b, v) in enumerate(zip(b0, v0)):
                    window[b.tobytes()].append(float(v[0]))
                    ply_of.setdefault(b.tobytes(), ply)
                if gid in test_ids:
                    for did in range(gid * aug, gid * aug + aug):
                        bb = f[f"boards_{did}"][:]
                        vv = f[f"values_{did}"][:]
                        for ply, (b, v) in enumerate(zip(bb, vv)):
                            k = b.tobytes()
                            test_pool[k].append(float(v[0]))
                            ply_of.setdefault(k, ply)

    # bucket-level per-sample sigma^2 from window repeats (fallback for sparse positions)
    bucket_var = {}
    for lo, hi in PLY_BUCKETS:
        vs = [np.var(v, ddof=1) for k, v in window.items()
              if lo <= ply_of[k] <= hi and len(v) >= 2]
        bucket_var[(lo, hi)] = np.mean(vs)

    def bvar(ply):
        for (lo, hi), v in bucket_var.items():
            if lo <= ply <= hi:
                return v
        return bucket_var[(16, 41)]

    S = sum(len(v) for v in test_pool.values())
    U = len(test_pool)
    once = sum(1 for v in test_pool.values() if len(v) == 1)
    floor = 0.0
    floor_b = {b: 0.0 for b in PLY_BUCKETS}
    n_b = {b: 0 for b in PLY_BUCKETS}
    for k, vals in test_pool.items():
        w = window.get(k, [])
        # mirror key may be absent from un-mirrored window dict - fall back to bucket
        s2 = np.var(w, ddof=1) if len(w) >= 3 else bvar(ply_of[k])
        floor += s2
        for lo, hi in PLY_BUCKETS:
            if lo <= ply_of[k] <= hi:
                floor_b[(lo, hi)] += s2
                n_b[(lo, hi)] += len(vals)
    floor /= S
    print(f"replicated TEST pool: {S} samples -> {U} unique, {once / U:.1%} of uniques once-only")
    print(f"EXACT val value-loss noise floor (perfect predictor): {floor:.4f}")
    print(f"observed best val value loss ~0.247 -> floor is {floor / 0.247:.0%} of it")
    for (lo, hi) in PLY_BUCKETS:
        sb = n_b[(lo, hi)]
        print(f"  bucket {lo}-{hi}: floor contribution {floor_b[(lo, hi)] / S:.4f} "
              f"({n_b[(lo,hi)]} samples)")

    # ---- empty-board drift significance ----
    print("\nempty-board per-gen target mean +/- stderr (z vs pooled mean):")
    empty = None
    for k, v in window.items():
        if ply_of[k] == 0:
            empty = k
            break
    per_gen = defaultdict(list)
    for g in GENS:
        with h5.File(f"Connect4/Grok_Zero_Train/{g}/Self_Play_Data.h5", "r") as f:
            n = int(f["game_stats"][2])
            aug = ((len(f.keys()) - 1) // 3) // n
            for gid in range(n):
                v = f[f"values_{gid * aug}"][:]
                per_gen[g].append(float(v[0][0]))
    allv = np.concatenate([per_gen[g] for g in GENS])
    pool_m = allv.mean()
    chi2 = 0.0
    for g in GENS:
        a = np.asarray(per_gen[g])
        se = a.std(ddof=1) / np.sqrt(len(a))
        z = (a.mean() - pool_m) / se
        chi2 += z * z
        print(f"  gen {g}: {a.mean():+.3f} +/- {se:.3f}  (n={len(a)}, z={z:+.1f})")
    print(f"  pooled {pool_m:+.3f}; chi^2(3) = {chi2:.0f} -> drift "
          f"{'REAL' if chi2 > 11.3 else 'not significant'} (p<0.01 cut 11.3)")


if __name__ == "__main__":
    main()
