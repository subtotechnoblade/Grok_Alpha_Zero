"""Compare arms at EQUAL POLICY QUALITY instead of equal epoch.

Epoch is not comparable across learning rates - a step at 3e-3 travels further than a
step at 3e-5, so "epoch 4" means something different in every arm. Policy KL is
monotone in training progress within an arm, so it works as a common x-axis: fix a
policy quality, ask what value head each arm delivers when it gets there. An arm that
wins at every level dominates outright; an arm that wins only at loose policy is just
trading one head for the other.

"-" means the arm never reached that policy quality at all.

Usage: python tmp/frontier.py [csv ...]
"""
import os
import sys
from collections import OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONST0 = 0.7691
LEVELS = (0.60, 0.57, 0.55, 0.53, 0.51, 0.50)


def load(path):
    rows = OrderedDict()
    with open(path) as f:
        head = f.readline().strip().split(",")
        for line in f:
            p = line.strip().split(",")
            if len(p) == len(head):
                rows.setdefault(p[0], []).append([float(x) for x in p[1:]])
    return head[1:], {k: np.array(v) for k, v in rows.items()}


def at_level(kl, val, target):
    """Value AT the epoch where this arm first reached `target` polKL.

    Must not be the running best: there is one shared checkpoint, so an early value
    head cannot be kept alongside a late policy head. Averaged over a +/-1 epoch
    window because the per-epoch value swings ~0.02 on its own.
    """
    reached = np.where(kl <= target)[0]
    if len(reached) == 0:
        return None
    n = reached[0]
    return float(val[max(0, n - 1):n + 2].mean())


def main():
    paths = sys.argv[1:] or [os.path.join(HERE, "exp5.csv"), os.path.join(HERE, "exp6.csv")]
    arms = OrderedDict()
    for p in paths:
        if not os.path.exists(p):
            continue
        cols, got = load(p)
        i = {c: n for n, c in enumerate(cols)}
        for k, v in got.items():
            arms[k] = (v[:, i["polKL"]], v[:, i["value"]], v[:, i["ep"]])
    if not arms:
        print("no csv data")
        return

    print("value head (% better than const-0) at matched policy quality")
    print("higher is better; an arm winning every column dominates outright\n")
    print(f"{'arm':<24} " + " ".join(f"{f'KL{l:.2f}':>8}" for l in LEVELS) + f" {'bestKL':>8}")
    print("-" * (25 + 9 * len(LEVELS) + 9))

    for tag, (kl, val, ep) in arms.items():
        cells = []
        for lv in LEVELS:
            v = at_level(kl, val, lv)
            cells.append("       -" if v is None else f"{1-v/CONST0:>7.1%}")
        print(f"{tag:<24} " + " ".join(cells) + f" {kl.min():>8.4f}")


if __name__ == "__main__":
    main()
