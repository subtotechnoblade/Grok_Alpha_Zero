"""Test the user's divergence hypothesis directly.

(A) DRIFT: for unique positions appearing in multiple generations, how much does
    their mean value target MOVE between generations, vs the outcome variance
    within one generation? If between-gen >> within-gen, the pooled 4-gen dedup
    average is teaching a smear of regimes - the value head chases a moving
    target and val loss cannot come down.
(B) FLOOR: fraction of positions occurring once (their +/-1 targets are pure
    variance), and within-gen outcome variance for repeated positions.
(C) LAG: gen-8 net's value MSE per source generation. Better on old gens than
    new = the net is calibrated to the stale majority of its window.

Read-only. Gens 4-7 only (8+ may be mid-write by the live run).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import h5py as h5
import onnxruntime as rt

rt.set_default_logger_severity(4)

GENS = [4, 5, 6, 7]
PLY_BUCKETS = [(0, 5), (6, 15), (16, 41)]


def iter_samples(gen):
    """Yield (state_bytes, ply, value_target, state_array) for the un-mirrored copy."""
    path = f"Connect4/Grok_Zero_Train/{gen}/Self_Play_Data.h5"
    with h5.File(path, "r") as f:
        n = int(f["game_stats"][2])
        n_sets = (len(f.keys()) - 1) // 3
        aug = n_sets // n
        for gid in range(n):
            did = gid * aug
            boards = f[f"boards_{did}"][:]
            values = f[f"values_{did}"][:]
            for ply, (b, v) in enumerate(zip(boards, values)):
                yield b.tobytes(), ply, float(v[0]), b


def main():
    # pass 1: per-gen per-position target means
    per_gen = {}          # gen -> {key: [targets]}
    ply_of = {}
    for g in GENS:
        d = {}
        for key, ply, v, _ in iter_samples(g):
            d.setdefault(key, []).append(v)
            ply_of.setdefault(key, ply)
        per_gen[g] = d
        n_once = sum(1 for v in d.values() if len(v) == 1)
        print(f"gen {g}: {sum(len(v) for v in d.values()):>6} samples, "
              f"{len(d):>6} unique, {n_once / len(d):.1%} occur once")

    print()
    print("(A) DRIFT vs (B) WITHIN-GEN VARIANCE, positions present in >=3 of gens 4-7")
    print(f"{'ply bucket':>12} {'n pos':>7} {'between-gen var':>16} "
          f"{'within-gen var':>15} {'ratio':>7}")
    common = [k for k in per_gen[GENS[0]]
              if sum(k in per_gen[g] for g in GENS) >= 3]
    for lo, hi in PLY_BUCKETS:
        rows_b, rows_w = [], []
        for k in common:
            if not (lo <= ply_of[k] <= hi):
                continue
            means, wvars = [], []
            for g in GENS:
                if k in per_gen[g]:
                    vals = np.array(per_gen[g][k])
                    means.append(vals.mean())
                    if len(vals) >= 3:
                        wvars.append(vals.var())
            if len(means) >= 3:
                rows_b.append(np.var(means))
                if wvars:
                    rows_w.append(np.mean(wvars))
        if rows_b:
            b, w = np.mean(rows_b), (np.mean(rows_w) if rows_w else float("nan"))
            print(f"{f'{lo}-{hi}':>12} {len(rows_b):>7} {b:>16.4f} {w:>15.4f} "
                  f"{b / w if w == w and w > 0 else float('nan'):>7.2f}")

    # (C) gen-8 net per-generation MSE
    print()
    print("(C) gen-8 net value MSE per source generation (5000 samples each)")
    sess = rt.InferenceSession("Connect4/Grok_Zero_Train/8/model.onnx",
                               providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    for g in GENS:
        X, y = [], []
        for key, ply, v, b in iter_samples(g):
            X.append(b)
            y.append(v)
        idx = rng.choice(len(X), size=min(5000, len(X)), replace=False)
        Xb = np.asarray([X[i] for i in idx], dtype=np.float32)
        yb = np.asarray([y[i] for i in idx], dtype=np.float32)
        preds = []
        for i in range(0, len(Xb), 512):
            _, val = sess.run(None, {"inputs": Xb[i:i + 512]})
            preds.append(val[:, 0])
        p = np.concatenate(preds)
        print(f"  gen {g}: MSE {float(np.mean((p - yb) ** 2)):.4f}")


if __name__ == "__main__":
    main()
