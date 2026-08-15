"""How much of the held-out error is irreducible label noise?

For a state s, self play gives outcomes drawn around the true expectation mu(s) with
variance sigma^2(s). dedup_average hands the learner the mean of the n occurrences,
so the label carries noise sigma^2(s)/n. Even a PERFECT model that outputs mu(s)
exactly therefore scores

    MSE_floor = E_w[ sigma^2(s) / n_val(s) ]

Nothing can beat that. If the best model is already near it, the value head is
noise-limited and more capacity / better optimisation cannot help.

sigma^2(s) is estimated from the TRAIN occurrences of s (unbiased sample variance),
so the estimate only exists for states seen >= 2 times in train. The comparison is
therefore restricted to the val states meeting that condition and is reported ON THAT
SUBSET for every quantity - no extrapolation to once-only states is attempted, and the
subset is explicitly NOT representative (it skews to openings). Read it as
"on positions where the noise level is measurable, how much of the error is noise?"
"""
import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import data
import features as F

MIN_N = 4   # train occurrences needed before we trust the variance estimate


def main(run="run2", gen=12, target="z"):
    print(f"### {run} g{gen} target={target}: irreducible label-noise floor\n")
    W = data.raw_window(run, gen, seed=0)
    tgt = "z" if target == "z" else "values"
    gs_tr = data.group_stats(W["train_games"], tgt)
    gs_va = data.group_stats(W["val_games"], tgt)

    ply_tr = (gs_tr["X"][:, :, :, 3] != 0).sum((1, 2))
    ok = gs_tr["count"] >= MIN_N
    print(f"  train: {len(gs_tr['X'])} unique, {ok.sum()} with >= {MIN_N} occurrences "
          f"(these carry a usable sigma^2 estimate)")
    print(f"  mean within-group sigma^2 by ply band (train, count>={MIN_N}):")
    for lo, hi in ((0, 5), (6, 9), (10, 13), (14, 19), (20, 41)):
        m = ok & (ply_tr >= lo) & (ply_tr <= hi)
        if m.sum() > 5:
            print(f"    ply {lo:2d}-{hi:2d}: n_groups={int(m.sum()):5d}  "
                  f"mean sigma^2={np.nanmean(gs_tr['var'][m]):.4f}  "
                  f"median count={int(np.median(gs_tr['count'][m]))}")

    # map val states -> train sigma^2
    tbl = {}
    for b, v, c in zip(gs_tr["X"], gs_tr["var"], gs_tr["count"]):
        if c >= MIN_N:
            tbl[b.tobytes()] = v
    sig = np.array([tbl.get(b.tobytes(), np.nan) for b in gs_va["X"]])
    have = ~np.isnan(sig)
    wsub = gs_va["w"][have]
    print(f"\n  val: {have.sum()}/{len(sig)} unique states have a measurable sigma^2 "
          f"({100*wsub.sum()/gs_va['w'].sum():.1f}% of val loss weight)")

    n_val = gs_va["count"][have]
    y = gs_va["mean"][have]
    noise_floor = float(np.sum(wsub * sig[have] / n_val) / np.sum(wsub))
    triv = float(np.sum(wsub * (y - 0.0) ** 2) / np.sum(wsub))

    # best available model, trained on the FULL train pool, scored on the same subset
    d = data.get(run, gen, seed=0)
    Xtr = F.VIEWS["hand+rel"](d["train"]["X"])
    m = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.06, max_leaf_nodes=63,
                                      l2_regularization=1.0, early_stopping=True,
                                      validation_fraction=0.15, n_iter_no_change=25,
                                      random_state=0).fit(Xtr, d["train"][f"y_{target}"],
                                                          sample_weight=d["train"]["w"])
    pred = m.predict(F.VIEWS["hand+rel"](gs_va["X"][have]))
    gbm = float(np.sum(wsub * (y - pred) ** 2) / np.sum(wsub))

    print(f"\n  ON THAT SUBSET ({int(have.sum())} val states, mean val count "
          f"{n_val.mean():.1f}, mean ply "
          f"{np.average((gs_va['X'][have][:, :, :, 3] != 0).sum((1, 2)), weights=wsub):.1f}):")
    print(f"    trivial floor (predict 0)          = {triv:.4f}")
    print(f"    HistGBM [hand+rel]                 = {gbm:.4f}")
    print(f"    irreducible label-noise floor      = {noise_floor:.4f}")
    print(f"    -> signal the model still misses   = {gbm - noise_floor:.4f} "
          f"({100*(gbm-noise_floor)/max(1e-9, triv-noise_floor):.1f}% of the learnable range)")
    print(f"    -> learnable range (triv - noise)  = {triv - noise_floor:.4f}")

    # what the floor WOULD be for once-only labels at the same sigma^2
    once = float(np.sum(wsub * sig[have]) / np.sum(wsub))
    print(f"\n    same states, if each label were a SINGLE game (n=1): floor = {once:.4f}")
    print(f"    (this is the regime {100*np.mean(d['val']['count']==1):.0f}% of val states are actually in)")

    out = dict(run=run, gen=gen, target=target, n_subset=int(have.sum()),
               weight_frac=float(wsub.sum() / gs_va["w"].sum()),
               trivial=triv, gbm=gbm, noise_floor=noise_floor, n1_floor=once)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"noise_{run}_g{gen}_{target}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "run2",
         int(sys.argv[2]) if len(sys.argv) > 2 else 12,
         sys.argv[3] if len(sys.argv) > 3 else "z")
