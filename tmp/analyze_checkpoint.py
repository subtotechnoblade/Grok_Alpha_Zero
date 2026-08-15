"""Which checkpoint criterion keeps training on the rails, across every arm ever run.

The claim under test: the criterion is what matters, and filters / dense width / lr /
optimizer / weight decay only change how fast and how well - so long as the criterion
never picks a checkpoint where the value head has gone bad. That makes this a
WORST-CASE question, not an average one. A criterion with a good mean and one
catastrophic pick is useless: the catastrophic pick is a poisoned generation.

criteria (each sees only what a Keras callback sees at epoch end)
  sum       polCE + value                        <- Train.py:60 val_loss today
  value     value only                           <- "best value regardless of policy"
  policy    policy only
  norm      value/value[ep1] + polCE/polCE[ep1]  <- scale-free
  vpat2     EarlyStopping(value, patience=2, restore_best_weights=True)
  sum_s     sum, 3-epoch centred moving average  <- value jitters ~0.02 epoch to epoch,
  norm_s    norm, same smoothing                    so the raw argmin partly picks luck

reference
  knee      max of (value gain fraction + policy gain fraction), each normalised to that
            arm's own observed range. Needs the whole curve, so it is not implementable
            online - it is the yardstick, not a candidate.

Usage: python tmp/analyze_checkpoint.py [csv ...]
"""
import os
import sys
from collections import OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONST0 = 0.7691
OFFRAIL = 0.05          # value less than 5% better than const-0 = a dead value head
DEFAULT = ["exp4.csv", "exp5.csv", "exp6.csv", "exp7.csv"]


def load(path):
    rows = OrderedDict()
    with open(path) as f:
        head = f.readline().strip().split(",")
        for line in f:
            p = line.strip().split(",")
            if len(p) == len(head):
                rows.setdefault(p[0], []).append([float(x) for x in p[1:]])
    return head[1:], {k: np.array(v) for k, v in rows.items()}


def smooth(x):
    return np.convolve(x, np.ones(3) / 3.0, mode="same")


def frac(x, lo, hi):
    return 1.0 if hi == lo else (lo - x) / (lo - hi)


def picks_for(v, ce, kl):
    """Epoch index each criterion selects. Index, not epoch number."""
    p = OrderedDict()
    p["sum"] = int((ce + v).argmin())
    p["sum_s"] = int(smooth(ce + v).argmin())
    p["norm"] = int((v / v[0] + ce / ce[0]).argmin())
    p["norm_s"] = int(smooth(v / v[0] + ce / ce[0]).argmin())
    p["value"] = int(v.argmin())
    p["policy"] = int(kl.argmin())
    best, wait, stop = v[0], 0, len(v) - 1
    for n in range(1, len(v)):
        if v[n] < best:
            best, wait = v[n], 0
        else:
            wait += 1
            if wait >= 2:
                stop = int(np.argmin(v[:n + 1]))
                break
    p["vpat2"] = stop
    return p


def main():
    paths = sys.argv[1:] or [os.path.join(HERE, f) for f in DEFAULT]
    arms = OrderedDict()
    for p in paths:
        if not os.path.exists(p):
            continue
        cols, got = load(p)
        i = {c: n for n, c in enumerate(cols)}
        for k, a in got.items():
            if len(a) >= 4:
                arms[k] = (a[:, i["ep"]], a[:, i["value"]], a[:, i["polCE"]], a[:, i["polKL"]])
    if not arms:
        print("no csv data")
        return

    names = list(picks_for(*[np.arange(1, 6, dtype=float)] * 3).keys())
    stats = {n: {"regret": [], "val": [], "pol": [], "off": 0} for n in names}

    print(f"per arm: epoch picked (value % better than const-0)   [{len(arms)} arms]\n")
    print(f"{'arm':<24} {'knee':>12} " + " ".join(f"{n:>12}" for n in names))
    print("-" * (25 + 13 * (len(names) + 1)))

    for tag, (ep, v, ce, kl) in arms.items():
        vlo, vhi, klo, khi = v.max(), v.min(), kl.max(), kl.min()
        joint = np.array([frac(v[n], vlo, vhi) + frac(kl[n], klo, khi) for n in range(len(ep))])
        knee = int(joint.argmax())
        p = picks_for(v, ce, kl)
        cells = [f"{int(ep[knee])}({1-v[knee]/CONST0:.0%})".rjust(12)]
        for n in names:
            k = p[n]
            stats[n]["regret"].append(joint[knee] - joint[k])
            stats[n]["val"].append(1 - v[k] / CONST0)
            stats[n]["pol"].append(frac(kl[k], klo, khi))
            stats[n]["off"] += int((1 - v[k] / CONST0) < OFFRAIL)
            cells.append(f"{int(ep[k])}({1-v[k]/CONST0:.0%})".rjust(12))
        print(f"{tag:<24} " + " ".join(cells))

    print(f"\nrobustness across all {len(arms)} arms "
          f"(off-rail = picked a checkpoint under {OFFRAIL:.0%} better than const-0)\n")
    print(f"{'criterion':<10} {'regret_mean':>12} {'regret_WORST':>13} "
          f"{'val_WORST':>10} {'polgain_WORST':>14} {'off-rail':>9}")
    print("-" * 72)
    for n in sorted(names, key=lambda n: max(stats[n]["regret"])):
        s = stats[n]
        print(f"{n:<10} {np.mean(s['regret']):>12.3f} {np.max(s['regret']):>13.3f} "
              f"{np.min(s['val']):>9.1%} {np.min(s['pol']):>13.1%} "
              f"{s['off']:>6}/{len(arms)}")


if __name__ == "__main__":
    main()
