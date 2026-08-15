"""All remaining discriminating experiments in ONE process, with cheap settings.

The machine is shared with unrelated jobs pinning it at 100% CPU, so the expensive
settings used in bench.py are traded down here: HistGBM max_iter 600->200 and
max_leaf_nodes 63->31, single-threaded, and the learning curve uses 2 seeds not 3.
That costs a little absolute accuracy (the cheap GBM scores ~0.01-0.02 worse than the
full one) but every number below is produced by the SAME cheap model, so the
comparisons between slices and between data volumes are internally consistent.
Anything read off against bench.py's table is flagged in the report.

Sections: 1 overlap  2 tempo-only  3 occurrence slices  4 noise floor  5 learning curve
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import data
import features as F

HERE = os.path.dirname(os.path.abspath(__file__))
GBM = dict(max_iter=200, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
           early_stopping=True, validation_fraction=0.15, n_iter_no_change=15)


def gbm(X, y, w, seed=0):
    return HistGradientBoostingRegressor(random_state=seed, **GBM).fit(X, y, sample_weight=w)


def wm(y, w):
    return float(np.sum(w * y) / np.sum(w))


def mse(y, p, w):
    return float(np.sum(w * (y - p) ** 2) / np.sum(w))


def main(run="run2", gen=12):
    t0 = time.time()
    out = {"run": run, "gen": gen, "gbm_settings": GBM}
    d = data.get(run, gen, seed=0)
    tr = d["train"]
    splits = [s for s in ("val", "nxt") if s in d]
    Xtr = F.VIEWS["hand+rel"](tr["X"])
    Xs = {s: F.VIEWS["hand+rel"](d[s]["X"]) for s in splits}
    train_keys = set(b.tobytes() for b in tr["X"])
    hit = {s: np.array([b.tobytes() in train_keys for b in d[s]["X"]]) for s in splits}
    ply = {s: (d[s]["X"][:, :, :, 3] != 0).sum((1, 2)) for s in splits}
    ply_tr = (tr["X"][:, :, :, 3] != 0).sum((1, 2))

    models = {}
    for target in ("blend", "z"):
        models[target] = gbm(Xtr, tr[f"y_{target}"], tr["w"])
    print(f"[fit cheap GBMs {time.time()-t0:.0f}s]", flush=True)

    # ---------------------------------------------------------------- 1 overlap
    print(f"\n=== 1. train/held-out position overlap ({run} g{gen}) ===")
    ov = []
    for s in splits:
        h = hit[s]
        wf = float(d[s]["w"][h].sum() / d[s]["w"].sum())
        print(f"  {s}: {h.sum()}/{len(h)} rows ({100*h.mean():.1f}%) byte-identical to a TRAIN "
              f"position = {100*wf:.1f}% of the split's loss WEIGHT "
              f"(mean ply {np.average(ply[s][h], weights=d[s]['w'][h]):.1f} vs "
              f"{np.average(ply[s][~h], weights=d[s]['w'][~h]):.1f})")
        ov.append({"split": s, "rows_frac": float(h.mean()), "weight_frac": wf})
    for target in ("blend", "z"):
        gm = wm(tr[f"y_{target}"], tr["w"])
        for s in splits:
            pr = models[target].predict(Xs[s])
            for nm, m in (("ALL", np.ones(len(hit[s]), bool)), ("seen-in-train", hit[s]),
                          ("truly-unseen", ~hit[s])):
                y, w = d[s][f"y_{target}"][m], d[s]["w"][m]
                e, fl = mse(y, pr[m], w), mse(y, np.full(m.sum(), gm), w)
                print(f"    {target:5s} {s:4s} {nm:13s} n={int(m.sum()):6d} floor={fl:.4f} "
                      f"GBM={e:.4f} R2={1-e/fl:+.4f}")
                ov.append({"target": target, "split": s, "subset": nm, "n": int(m.sum()),
                           "floor": fl, "gbm": e, "r2": 1 - e / fl})
    out["overlap"] = ov

    # ---------------------------------------------------------------- 2 tempo only
    print(f"\n=== 2. tempo-only baselines (no board information) ===")
    tv = []
    for target in ("blend", "z"):
        y, w = tr[f"y_{target}"], tr["w"]
        gm = wm(y, w)
        side_tr = -tr["X"][:, 0, 0, 0].astype(np.int64)
        for nm, ktr, kev in (
                ("per-ply mean", list(zip(ply_tr.tolist())),
                 lambda s: list(zip(ply[s].tolist()))),
                ("per-(ply,side) mean", list(zip(ply_tr.tolist(), side_tr.tolist())),
                 lambda s: list(zip(ply[s].tolist(),
                                    (-d[s]["X"][:, 0, 0, 0].astype(np.int64)).tolist())))):
            num, den = {}, {}
            for k, yy, ww in zip(ktr, y, w):
                num[k] = num.get(k, 0.0) + ww * yy
                den[k] = den.get(k, 0.0) + ww
            tbl = {k: num[k] / den[k] for k in num}
            row = {"model": nm, "target": target}
            for s in splits:
                pr = np.array([tbl.get(k, gm) for k in kev(s)])
                row[s] = mse(d[s][f"y_{target}"], pr, d[s]["w"])
            tv.append(row)
            print(f"  {nm:24s} {target:5s} " + "  ".join(f"{s}={row[s]:.4f}" for s in splits))
    out["tempo"] = tv

    # ---------------------------------------------------------------- 3 occurrence
    print(f"\n=== 3. signal by occurrence count (label noise ~ 1/n) ===")
    oc = []
    for target in ("blend", "z"):
        gm = wm(tr[f"y_{target}"], tr["w"])
        print(f"  -- {target}: same model, held-out sliced by the held-out row's count --")
        for s in splits:
            pr = models[target].predict(Xs[s])
            for lo, hi, nm in ((1, 1, "==1"), (2, 4, "2-4"), (5, 9, "5-9"), (10, 10**9, ">=10")):
                m = (d[s]["count"] >= lo) & (d[s]["count"] <= hi)
                if m.sum() < 20:
                    continue
                y, w = d[s][f"y_{target}"][m], d[s]["w"][m]
                e, fl = mse(y, pr[m], w), mse(y, np.full(m.sum(), gm), w)
                print(f"    {s:4s} count {nm:4s} n={int(m.sum()):6d} meanply="
                      f"{np.average(ply[s][m], weights=w):4.1f}  floor={fl:.4f} GBM={e:.4f} "
                      f"R2={1-e/fl:+.4f}")
                oc.append({"target": target, "split": s, "band": nm, "phase": "eval-sliced",
                           "n": int(m.sum()), "floor": fl, "gbm": e, "r2": 1 - e / fl})
        print(f"  -- {target}: ply-CONTROLLED (removes the opening/endgame confound) --")
        for band in ((6, 13), (14, 21)):
            bt = (ply_tr >= band[0]) & (ply_tr <= band[1])
            if bt.sum() < 500:
                continue
            mb = gbm(Xtr[bt], tr[f"y_{target}"][bt], tr["w"][bt])
            gml = wm(tr[f"y_{target}"][bt], tr["w"][bt])
            for s in splits:
                bb = (ply[s] >= band[0]) & (ply[s] <= band[1])
                if bb.sum() < 40:
                    continue
                pr = mb.predict(Xs[s][bb])
                for lo, hi, nm in ((1, 1, "count==1"), (2, 10**9, "count>=2")):
                    sub = (d[s]["count"][bb] >= lo) & (d[s]["count"][bb] <= hi)
                    if sub.sum() < 20:
                        continue
                    y, w = d[s][f"y_{target}"][bb][sub], d[s]["w"][bb][sub]
                    e, fl = mse(y, pr[sub], w), mse(y, np.full(sub.sum(), gml), w)
                    print(f"    {s:4s} ply {band[0]:2d}-{band[1]:2d} {nm:9s} n={int(sub.sum()):6d} "
                          f"floor={fl:.4f} GBM={e:.4f} R2={1-e/fl:+.4f}")
                    oc.append({"target": target, "split": s, "band": f"ply{band[0]}-{band[1]}/{nm}",
                               "phase": "ply-controlled", "n": int(sub.sum()), "floor": fl,
                               "gbm": e, "r2": 1 - e / fl})
    out["occurrence"] = oc
    print(f"[{time.time()-t0:.0f}s]", flush=True)

    # ---------------------------------------------------------------- 4+5 need raw games
    print(f"\n=== 4. irreducible label-noise floor + 5. learning curve ===", flush=True)
    W = data.raw_window(run, gen, seed=0)
    games, val_p, nxt_p = W["train_games"], W["val"], W.get("nxt")
    nf = []
    for target in ("blend", "z"):
        tgt = "z" if target == "z" else "values"
        gt = data.group_stats(W["train_games"], tgt)
        gv = data.group_stats(W["val_games"], tgt)
        MIN_N = 4
        tbl = {b.tobytes(): v for b, v, c in zip(gt["X"], gt["var"], gt["count"]) if c >= MIN_N}
        sig = np.array([tbl.get(b.tobytes(), np.nan) for b in gv["X"]])
        have = ~np.isnan(sig)
        w = gv["w"][have]
        n_val = gv["count"][have]
        y = gv["mean"][have]
        floor_noise = float(np.sum(w * sig[have] / n_val) / np.sum(w))
        n1 = float(np.sum(w * sig[have]) / np.sum(w))
        triv = float(np.sum(w * y ** 2) / np.sum(w))
        pr = models[target].predict(F.VIEWS["hand+rel"](gv["X"][have]))
        g = float(np.sum(w * (y - pr) ** 2) / np.sum(w))
        print(f"  {target}: subset = {int(have.sum())}/{len(sig)} val states with a measurable "
              f"sigma^2 ({100*w.sum()/gv['w'].sum():.1f}% of val weight, mean n_val "
              f"{n_val.mean():.1f}, mean ply "
              f"{np.average((gv['X'][have][:,:,:,3]!=0).sum((1,2)), weights=w):.1f})")
        print(f"     trivial={triv:.4f}  GBM={g:.4f}  irreducible-noise floor={floor_noise:.4f}  "
              f"| same states with n=1 labels the floor would be {n1:.4f}")
        print(f"     -> of the {triv-floor_noise:.4f} learnable range, GBM still misses "
              f"{g-floor_noise:.4f} ({100*(g-floor_noise)/max(1e-9,triv-floor_noise):.0f}%)")
        nf.append({"target": target, "n_subset": int(have.sum()),
                   "weight_frac": float(w.sum() / gv["w"].sum()), "trivial": triv, "gbm": g,
                   "noise_floor": floor_noise, "n1_floor": n1})
    out["noise_floor"] = nf
    print(f"[{time.time()-t0:.0f}s]", flush=True)

    print(f"\n  learning curve ({len(games)} training games in the window), by GAME:")
    lc = []
    for frac in (0.10, 0.25, 0.50, 1.00):
        n = max(10, int(round(frac * len(games))))
        for seed in (0, 1):
            rng = np.random.RandomState(1000 + seed)
            pool = data.pack_games([games[i] for i in rng.choice(len(games), n, replace=False)])
            Xp = F.VIEWS["hand+rel"](pool["X"])
            for target in ("blend", "z"):
                m = gbm(Xp, pool[f"y_{target}"], pool["w"], seed)
                r = {"model": "HistGBM(cheap)", "target": target, "frac": frac, "n_games": n,
                     "seed": seed, "n_unique": int(len(pool["X"])), "n_raw": int(pool["n_raw"]),
                     "train": mse(pool[f"y_{target}"], m.predict(Xp), pool["w"]),
                     "val": mse(val_p[f"y_{target}"],
                                m.predict(F.VIEWS["hand+rel"](val_p["X"])), val_p["w"])}
                if nxt_p is not None:
                    r["nxt"] = mse(nxt_p[f"y_{target}"],
                                   m.predict(F.VIEWS["hand+rel"](nxt_p["X"])), nxt_p["w"])
                lc.append(r)
                print(f"    {target:5s} frac={frac:.2f} games={n:5d} unique={r['n_unique']:6d} "
                      f"seed={seed} train={r['train']:.4f} val={r['val']:.4f} "
                      f"nxt={r.get('nxt', float('nan')):.4f}", flush=True)
    out["learning_curve"] = lc
    print("\n  curve summary (mean over 2 seeds):")
    for target in ("blend", "z"):
        for frac in (0.10, 0.25, 0.50, 1.00):
            sub = [r for r in lc if r["frac"] == frac and r["target"] == target]
            v = np.array([r["val"] for r in sub]); nx = np.array([r["nxt"] for r in sub])
            print(f"    {target:5s} frac={frac:.2f} games={sub[0]['n_games']:5d} "
                  f"unique={sub[0]['n_unique']:6d}  val={v.mean():.4f} (+-{(v.max()-v.min())/2:.4f})"
                  f"  nxt={nx.mean():.4f} (+-{(nx.max()-nx.min())/2:.4f})")

    p = os.path.join(HERE, f"fast_{run}_g{gen}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {p} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "run2",
         int(sys.argv[2]) if len(sys.argv) > 2 else 12)
