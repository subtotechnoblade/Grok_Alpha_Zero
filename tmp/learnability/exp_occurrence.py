"""Experiment A: is the value target NOISE-limited?

If a position was seen n times, its dedup-averaged label is a mean of n game outcomes,
so its label noise falls like 1/n. Slice the SAME held-out set by occurrence count and
compare each slice's model error to that slice's OWN trivial floor.

  R2 = 1 - MSE_model / MSE_floor(predict the train weighted mean)

CONFOUND, stated up front: occurrence count correlates almost perfectly with game
phase (opening positions repeat, endgames do not). So a raw count slice mixes "less
label noise" with "easier position". The ply-stratified section below controls for it:
inside one ply band, compare high-count against low-count positions.
"""
import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import data
import features as F

SEED = 0


def fit_gbm(X, y, w, seed=SEED):
    return HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.06, max_leaf_nodes=63, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=25,
        random_state=seed).fit(X, y, sample_weight=w)


def report(tag, y, p, w, floor_pred, rows, extra=None):
    mse = data.wmse(y, p, w)
    fl = data.wmse(y, np.full(len(y), floor_pred), w)
    r2 = 1 - mse / fl if fl > 0 else float("nan")
    r = {"slice": tag, "n": int(len(y)), "mse": mse, "floor": fl, "r2": r2}
    if extra:
        r.update(extra)
    rows.append(r)
    print(f"  {tag:44s} n={len(y):6d}  floor={fl:.4f}  mse={mse:.4f}  R2={r2:+.4f}")
    return r


def main(run="run2", gen=12, target="blend"):
    d = data.get(run, gen, seed=SEED)
    tr = d["train"]
    splits = [s for s in ("val", "nxt") if s in d]
    Xtr_all = F.VIEWS["hand+rel"](tr["X"])
    ytr_all = tr[f"y_{target}"]
    gmean = float(np.sum(tr["w"] * ytr_all) / np.sum(tr["w"]))
    rows = []

    print(f"### {run} g{gen} target={target}: occurrence-count slices\n")
    print("A1. train on ALL train data, evaluate on held-out slices by the HELD-OUT "
          "position's own occurrence count")
    m_all = fit_gbm(Xtr_all, ytr_all, tr["w"])
    for s in splits:
        p = d[s]
        pred = m_all.predict(F.VIEWS["hand+rel"](p["X"]))
        for lo, hi, nm in ((1, 1, "==1"), (2, 4, "2-4"), (5, 9, "5-9"), (10, 10**9, ">=10")):
            m = (p["count"] >= lo) & (p["count"] <= hi)
            if m.sum() < 20:
                continue
            report(f"{s} count {nm}", p[f"y_{target}"][m], pred[m], p["w"][m], gmean, rows,
                   {"split": s, "phase": "A1", "mean_ply": float(
                       np.average((p["X"][m][:, :, :, 3] != 0).sum((1, 2)), weights=p["w"][m]))})

    print("\nA2. train AND evaluate restricted to the same occurrence band "
          "(literal version; small-n warning where noted)")
    for lo, hi, nm in ((1, 1, "==1"), (2, 4, "2-4"), (5, 9, "5-9"), (10, 10**9, ">=10")):
        mt = (tr["count"] >= lo) & (tr["count"] <= hi)
        if mt.sum() < 200:
            print(f"  [skip train count {nm}: only {mt.sum()} train rows]")
            continue
        mloc = float(np.sum(tr["w"][mt] * ytr_all[mt]) / np.sum(tr["w"][mt]))
        m = fit_gbm(Xtr_all[mt], ytr_all[mt], tr["w"][mt])
        for s in splits:
            p = d[s]
            me = (p["count"] >= lo) & (p["count"] <= hi)
            if me.sum() < 20:
                continue
            pred = m.predict(F.VIEWS["hand+rel"](p["X"][me]))
            report(f"{s} count {nm} (trained on {int(mt.sum())})",
                   p[f"y_{target}"][me], pred, p["w"][me], mloc, rows,
                   {"split": s, "phase": "A2", "n_train": int(mt.sum())})

    print("\nA3. ply-controlled: inside one ply band, high-count vs low-count "
          "(removes the phase confound)")
    ply_tr = (tr["X"][:, :, :, 3] != 0).sum((1, 2))
    for band in ((0, 7), (8, 13), (14, 21)):
        bt = (ply_tr >= band[0]) & (ply_tr <= band[1])
        if bt.sum() < 500:
            continue
        mb = fit_gbm(Xtr_all[bt], ytr_all[bt], tr["w"][bt])
        mloc = float(np.sum(tr["w"][bt] * ytr_all[bt]) / np.sum(tr["w"][bt]))
        for s in splits:
            p = d[s]
            ply = (p["X"][:, :, :, 3] != 0).sum((1, 2))
            bb = (ply >= band[0]) & (ply <= band[1])
            if bb.sum() < 40:
                continue
            pred = mb.predict(F.VIEWS["hand+rel"](p["X"][bb]))
            for lo, hi, nm in ((1, 1, "count==1"), (2, 10**9, "count>=2")):
                mm = bb.copy()
                mm[bb] = (p["count"][bb] >= lo) & (p["count"][bb] <= hi)
                if mm.sum() < 20:
                    continue
                sub = mm[bb]
                report(f"{s} ply {band[0]}-{band[1]} {nm}", p[f"y_{target}"][mm], pred[sub],
                       p["w"][mm], mloc, rows,
                       {"split": s, "phase": "A3", "band": f"{band[0]}-{band[1]}"})

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"occurrence_{run}_g{gen}_{target}.json")
    with open(path, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "run2",
         int(sys.argv[2]) if len(sys.argv) > 2 else 12,
         sys.argv[3] if len(sys.argv) > 3 else "blend")
