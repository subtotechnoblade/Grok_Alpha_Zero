"""Where does the training gradient mass actually go, and what would clamping change?

dedup_average gives each unique position a sample_weight equal to its occurrence count.
That is gradient-identical to duplicating it, which is the point. But it also means the
opening - which occurs in every game - carries enormous weight while near-terminal
positions occur once each.

Prints the weight distribution, the share of total gradient mass per plies-to-end bucket,
and what a clamp would do to that share.
"""
import os
import sys

import numpy as np
import h5py as h5

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(HERE))

RUN = "Connect4/Grok_Zero_Train"
GENS = [int(g) for g in (sys.argv[1:] or ["1", "2", "3"])]
BUCKETS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 11), (12, 15), (16, 23), (24, 99)]
CLAMPS = [None, 100, 50, 20, 10, 5, 1]


def load(gens):
    B, V, R = [], [], []
    for g in gens:
        with h5.File(f"{RUN}/{g}/Self_Play_Data.h5", "r") as f:
            n = int(f["game_stats"][2])
            aug = ((len(f.keys()) - 1) // 3) // n
            for gi in range(n):
                for a in range(aug):
                    i = gi * aug + a
                    b = f[f"boards_{i}"][:]
                    B.append(b)
                    V.append(f[f"values_{i}"][:][:, 0])
                    R.append(np.arange(len(b))[::-1])
    return np.concatenate(B), np.concatenate(V), np.concatenate(R)


def main():
    X, V, R = load(GENS)
    order = {}
    for i, x in enumerate(X):
        order.setdefault(x.tobytes(), []).append(i)
    idx = np.array([v[0] for v in order.values()])
    cnt = np.array([len(v) for v in order.values()], dtype=np.float64)
    tgt = np.array([V[v].mean() for v in order.values()])
    rem = np.array([R[v].mean() for v in order.values()])

    w = cnt * (len(cnt) / cnt.sum())
    ess = w.sum() ** 2 / (w ** 2).sum()
    print(f"gens {GENS}: {len(X)} raw positions -> {len(cnt)} unique ({len(cnt)/len(X):.1%})")
    print(f"occurrence count: max {cnt.max():.0f}  p99 {np.percentile(cnt,99):.0f}  "
          f"median {np.median(cnt):.0f}  mean {cnt.mean():.2f}")
    print(f"effective sample size {ess:.0f} of {len(cnt)} unique "
          f"({ess/len(cnt):.1%}) - weighting throws away this much of the dataset\n")

    print("share of total gradient mass by plies-to-end, under different clamps")
    print(f"{'plies':>9} {'uniq':>7} " + " ".join(
        f"{('none' if c is None else f'<={c}'):>7}" for c in CLAMPS))
    print("-" * (18 + 8 * len(CLAMPS)))
    masks = [(lo, hi, (rem >= lo) & (rem <= hi)) for lo, hi in BUCKETS]
    for lo, hi, m in masks:
        cells = []
        for c in CLAMPS:
            wc = cnt if c is None else np.minimum(cnt, c)
            cells.append(f"{wc[m].sum()/wc.sum():>6.1%}")
        print(f"{f'{lo}-{hi}':>9} {m.sum():>7} " + " ".join(cells))

    print(f"\n{'':>9} {'':>7} " + " ".join(
        f"{('none' if c is None else f'<={c}'):>7}" for c in CLAMPS))
    row = []
    for c in CLAMPS:
        wc = cnt if c is None else np.minimum(cnt, c)
        row.append(f"{(wc.sum()**2/(wc**2).sum())/len(cnt):>6.1%}")
    print(f"{'eff N':>9} {'':>7} " + " ".join(row))
    row = []
    for c in CLAMPS:
        wc = cnt if c is None else np.minimum(cnt, c)
        row.append(f"{np.average(np.abs(tgt), weights=wc):>6.3f}")
    print(f"{'w|target|':>9} {'':>7} " + " ".join(row))
    row = []
    for c in CLAMPS:
        wc = cnt if c is None else np.minimum(cnt, c)
        dec = np.abs(tgt) > 0.5
        row.append(f"{wc[dec].sum()/wc.sum():>6.1%}")
    print(f"{'decisive%':>9} {'':>7} " + " ".join(row))

    top = np.argsort(-cnt)[:5]
    print("\nheaviest positions:")
    for t in top:
        print(f"  count {cnt[t]:>6.0f}  weight {w[t]:>7.1f}  "
              f"mean plies-to-end {rem[t]:>5.1f}  target {tgt[t]:+.3f}")


if __name__ == "__main__":
    main()
