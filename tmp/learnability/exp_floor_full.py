"""The noise floor for the WHOLE val split, using the measured sigma^2(ply) curve.

exp_fast section 4 could only measure the floor on positions repeated >= 4 times, and
exp_sigma_ply showed those sit at ply ~10 while sigma^2 collapses from ~0.72 (ply 0-5)
to ~0.00 (ply 30+). Applying the ply-10 noise level to every position would badly
overstate the floor. Here sigma^2 is estimated PER PLY from the count>=2 groups and
applied to each val state at its own ply and its own occurrence count:

    MSE_floor = sum_i w_i * sigma^2(ply_i) / n_i  /  sum_i w_i

KNOWN BIAS, not hidden: sigma^2(ply) is measured only on positions that repeat, which
at a given ply are more "mainline" than a one-off. The direction of that bias is not
established, so treat the floor as an estimate with maybe +-25% of slack, not a
constant. The qualitative gap it reveals is far larger than that slack.
"""
import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import data
import features as F

BANDS = [(0, 5), (6, 9), (10, 13), (14, 17), (18, 21), (22, 29), (30, 41)]


def main(run="run2", gen=12):
    W = data.raw_window(run, gen, seed=0)
    d = data.get(run, gen, seed=0)
    tr = d["train"]
    Xtr = F.VIEWS["hand+rel"](tr["X"])
    out = {"run": run, "gen": gen, "bands": [f"{a}-{b}" for a, b in BANDS]}
    print(f"### {run} g{gen}: ply-aware irreducible noise floor over the FULL held-out split\n")

    for target in ("blend", "z"):
        tgt = "z" if target == "z" else "values"
        gt = data.group_stats(W["train_games"], tgt)
        ply_g = (gt["X"][:, :, :, 3] != 0).sum((1, 2))
        sig = {}
        for lo, hi in BANDS:
            m = (gt["count"] >= 2) & (ply_g >= lo) & (ply_g <= hi)
            sig[(lo, hi)] = float(np.nanmean(gt["var"][m])) if m.sum() >= 10 else 0.0

        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
                                          l2_regularization=1.0, early_stopping=True,
                                          validation_fraction=0.15, n_iter_no_change=15,
                                          random_state=0).fit(Xtr, tr[f"y_{target}"],
                                                              sample_weight=tr["w"])
        print(f"  -- target={target} --")
        rows = []
        for s in ("val", "nxt"):
            if s not in d:
                continue
            p = d[s]
            ply = (p["X"][:, :, :, 3] != 0).sum((1, 2))
            s2 = np.zeros(len(ply))
            for (lo, hi), v in sig.items():
                s2[(ply >= lo) & (ply <= hi)] = v
            w = p["w"]
            floor = float(np.sum(w * s2 / p["count"]) / np.sum(w))
            triv = float(np.sum(w * p[f"y_{target}"] ** 2) / np.sum(w))
            pr = m.predict(F.VIEWS["hand+rel"](p["X"]))
            g = float(np.sum(w * (p[f"y_{target}"] - pr) ** 2) / np.sum(w))
            print(f"    {s}: trivial(predict 0)={triv:.4f}  cheap-GBM={g:.4f}  "
                  f"est. noise floor={floor:.4f}")
            print(f"       learnable range = {triv-floor:.4f}; GBM captures "
                  f"{100*(triv-g)/max(1e-9, triv-floor):.0f}% of it, "
                  f"{g-floor:.4f} MSE still on the table")
            rows.append({"split": s, "trivial": triv, "gbm": g, "floor": floor})
        out[target] = {"sigma2_by_band": {f"{a}-{b}": sig[(a, b)] for a, b in BANDS},
                       "rows": rows}
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"floorfull_{run}_g{gen}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "run2",
         int(sys.argv[2]) if len(sys.argv) > 2 else 12)
