"""How much of the value signal is TEMPO, not position?

A per-ply (and per-ply-per-side) group mean uses no board information at all beyond
"how far into the game are we and whose turn is it". If a 400k-parameter ResNet is
close to this, it has learned essentially nothing about the position itself.
Groups are fit on train only, with the production occurrence weights; unseen groups
fall back to the global train weighted mean.
"""
import json
import os
import sys

import numpy as np

import data
import features as F


def group_predictor(keys_tr, y, w, keys_ev, fallback):
    num, den = {}, {}
    for k, yy, ww in zip(keys_tr, y, w):
        num[k] = num.get(k, 0.0) + ww * yy
        den[k] = den.get(k, 0.0) + ww
    tbl = {k: num[k] / den[k] for k in num}
    return np.array([tbl.get(k, fallback) for k in keys_ev])


def ply_of(X):
    return (X[:, :, :, 3] != 0).sum((1, 2)).astype(np.int64)


def main(run="run2", gen=12):
    d = data.get(run, gen, seed=0)
    tr = d["train"]
    splits = [s for s in ("val", "nxt") if s in d]
    rows = []
    print(f"### {run} g{gen}: tempo-only baselines\n")
    for target in ("blend", "z"):
        y = tr[f"y_{target}"]
        gm = float(np.sum(tr["w"] * y) / np.sum(tr["w"]))
        ply_tr = ply_of(tr["X"])
        side_tr = -tr["X"][:, 0, 0, 0].astype(np.int64)
        h_tr = F.hand(tr["X"])
        i_my3 = F.HAND_NAMES.index("my_playable3")
        i_op3 = F.HAND_NAMES.index("opp_playable3")
        thr_tr = (np.clip(h_tr[:, i_my3], 0, 2).astype(np.int64) * 3 +
                  np.clip(h_tr[:, i_op3], 0, 2).astype(np.int64))

        defs = {
            "per-ply mean": (list(zip(ply_tr.tolist())), lambda p: list(zip(ply_of(p["X"]).tolist()))),
            "per-(ply,side) mean": (
                list(zip(ply_tr.tolist(), side_tr.tolist())),
                lambda p: list(zip(ply_of(p["X"]).tolist(),
                                   (-p["X"][:, 0, 0, 0].astype(np.int64)).tolist()))),
            "per-(ply,side,playable-threats) mean": (
                list(zip(ply_tr.tolist(), side_tr.tolist(), thr_tr.tolist())),
                None),
        }
        for name, (ktr, kev) in defs.items():
            row = {"model": name, "target": target}
            for s in splits:
                p = d[s]
                if kev is None:
                    h = F.hand(p["X"])
                    t = (np.clip(h[:, i_my3], 0, 2).astype(np.int64) * 3 +
                         np.clip(h[:, i_op3], 0, 2).astype(np.int64))
                    ke = list(zip(ply_of(p["X"]).tolist(),
                                  (-p["X"][:, 0, 0, 0].astype(np.int64)).tolist(), t.tolist()))
                else:
                    ke = kev(p)
                pr = group_predictor(ktr, y, tr["w"], ke, gm)
                row[s] = data.wmse(p[f"y_{target}"], pr, p["w"])
            rows.append(row)
            print(f"  {name:38s} {target:5s} " + "  ".join(f"{s}={row[s]:.4f}" for s in splits))
        print()

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"trivial_{run}_g{gen}.json")
    with open(path, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "run2",
         int(sys.argv[2]) if len(sys.argv) > 2 else 12)
