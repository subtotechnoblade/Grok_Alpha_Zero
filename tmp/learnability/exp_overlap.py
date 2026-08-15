"""Why production's val_loss flatters the value head.

Dataloader splits by GAME, which stops a board and its own mirror landing on both
sides - but it does NOT stop the same opening position being reached by a train game
and a val game. dedup_average runs per side, so those byte-identical states survive
as one train row and one val row. Anything that memorises train therefore scores on
a large, heavily-weighted chunk of val for free.

This measures the size of that chunk and splits every error number by it.
"""
import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import data
import features as F


def main(run="run2", gen=12):
    d = data.get(run, gen, seed=0)
    tr = d["train"]
    splits = [s for s in ("val", "nxt") if s in d]
    train_keys = set(b.tobytes() for b in tr["X"])
    ply_tr = (tr["X"][:, :, :, 3] != 0).sum((1, 2))
    out = []
    print(f"### {run} g{gen}: train/held-out position overlap\n")
    for s in splits:
        p = d[s]
        hit = np.array([b.tobytes() in train_keys for b in p["X"]])
        wfrac = float(p["w"][hit].sum() / p["w"].sum())
        ply = (p["X"][:, :, :, 3] != 0).sum((1, 2))
        print(f"  {s}: {hit.sum()}/{len(hit)} rows ({100*hit.mean():.1f}%) are byte-identical to a "
              f"TRAIN position -> {100*wfrac:.1f}% of the split's loss WEIGHT")
        print(f"      mean ply of overlapping {np.average(ply[hit], weights=p['w'][hit]):.1f} "
              f"vs non-overlapping {np.average(ply[~hit], weights=p['w'][~hit]):.1f}")
        out.append({"split": s, "run": run, "gen": gen, "rows_frac": float(hit.mean()),
                    "weight_frac": wfrac})

    Xtr = F.VIEWS["hand+rel"](tr["X"])
    for target in ("blend", "z"):
        print(f"\n  -- target={target}: error split by overlap --")
        y = tr[f"y_{target}"]
        gm = float(np.sum(tr["w"] * y) / np.sum(tr["w"]))
        m = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.06, max_leaf_nodes=63,
                                          l2_regularization=1.0, early_stopping=True,
                                          validation_fraction=0.15, n_iter_no_change=25,
                                          random_state=0).fit(Xtr, y, sample_weight=tr["w"])
        for s in splits:
            p = d[s]
            hit = np.array([b.tobytes() in train_keys for b in p["X"]])
            pr = m.predict(F.VIEWS["hand+rel"](p["X"]))
            yy = p[f"y_{target}"]
            for nm, mask in (("ALL", np.ones(len(hit), bool)), ("seen-in-train", hit),
                             ("truly-unseen", ~hit)):
                if mask.sum() < 5:
                    continue
                mse = data.wmse(yy[mask], pr[mask], p["w"][mask])
                fl = data.wmse(yy[mask], np.full(mask.sum(), gm), p["w"][mask])
                print(f"    {s:4s} {nm:14s} n={int(mask.sum()):6d}  floor={fl:.4f}  "
                      f"GBM={mse:.4f}  R2={1-mse/fl:+.4f}")
                out.append({"split": s, "target": target, "subset": nm, "n": int(mask.sum()),
                            "floor": fl, "gbm": mse, "r2": 1 - mse / fl})

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"overlap_{run}_g{gen}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "run2",
         int(sys.argv[2]) if len(sys.argv) > 2 else 12)
