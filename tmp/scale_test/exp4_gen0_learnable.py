"""EXP 4 (standalone, no TF) - is generation-0 data learnable AT ALL?

The exp3 arms all used the same 600-iteration / 63-leaf GBM that the rest of this
folder uses.  On a pool that is 97% singletons that model is guaranteed to overfit,
so "GBM loses to predicting zero" would not by itself prove the data is signal-free.
This script therefore sweeps capacity from a 1-parameter constant up to the big GBM,
including heavily regularised arms, and reports every one against const-0.

If even the best-regularised arm cannot beat const-0 on a held-out set of gen-0
games, the conclusion is about the DATA, not about the model.

Diagnostics printed first: occurrence counts, singleton share, and the within-group
variance of z per ply band - the part of the label that averaging could remove if
positions repeated, and which is undefined (single sample) where they do not.
"""
import sys

import numpy as np

import common as C

TARGET = sys.argv[1] if len(sys.argv) > 1 else "z"
SEEDS = (0, 1, 2)
P = print


def diagnostics(run):
    games = C.read_gen(run, 0)
    p = C.pack(games)
    gs = C.D.group_stats(games, target="z")
    ply = C.ply_of(p["X"])
    plyg = C.ply_of(gs["X"])
    P(f"\n--- {run} gen 0: {len(games)} games, {p['n_raw']} raw -> {len(p['X'])} unique ---",
      flush=True)
    P(f"  singletons: {100*np.mean(p['count']==1):.1f}% of unique rows, "
      f"{100*p['w'][p['count']==1].sum()/p['w'].sum():.1f}% of loss weight", flush=True)
    P(f"  occurrence count: mean {p['count'].mean():.2f}  median {np.median(p['count']):.0f}  "
      f"p99 {np.percentile(p['count'],99):.0f}  max {p['count'].max()}", flush=True)
    P(f"  {'ply band':>9} {'uniq':>7} {'wt%':>6} {'mean cnt':>9} {'singles%':>9} "
      f"{'var(z) in-group':>16}", flush=True)
    for lo, hi in ((0, 5), (6, 9), (10, 13), (14, 17), (18, 21), (22, 29), (30, 41)):
        m = (ply >= lo) & (ply <= hi)
        if not m.sum():
            continue
        vv = gs["var"][(plyg >= lo) & (plyg <= hi)]
        vv = vv[~np.isnan(vv)]
        var = f"{vv.mean():.4f}" if len(vv) else "n/a (all single)"
        P(f"  {f'{lo}-{hi}':>9} {int(m.sum()):>7} {100*p['w'][m].sum()/p['w'].sum():>5.1f}% "
          f"{p['count'][m].mean():>9.2f} {100*np.mean(p['count'][m]==1):>8.1f}% {var:>16}",
          flush=True)
    return games


def models(pool, evals, target, seed):
    """Capacity ladder from 1 parameter to the production-sized GBM."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    import features as F

    y = pool[f"y_{target}"]
    w = pool["w"]
    out = []

    def add(name, pred_fn, note=""):
        r = {"model": name, "note": note, "seed": seed,
             "train": C.wmse(y, pred_fn(pool), w)}
        for nm, p in evals.items():
            r[nm] = C.wmse(p[f"y_{target}"], pred_fn(p), p["w"])
        out.append(r)
        return r

    add("const 0", lambda p: np.zeros(len(p["X"])))
    mu = float(np.sum(w * y) / np.sum(w))
    add("const train mean", lambda p, mu=mu: np.full(len(p["X"]), mu), f"mu={mu:.4f}")

    Xh = F.VIEWS["hand"](pool["X"])
    sc = StandardScaler().fit(Xh)
    for a in (1e2, 1e3, 1e4, 1e5):
        m = Ridge(alpha=a).fit(sc.transform(Xh), y, sample_weight=w)
        add(f"ridge[hand] a={a:g}",
            lambda p, m=m, sc=sc: m.predict(sc.transform(F.VIEWS["hand"](p["X"]))))

    for tag, kw in (("tiny", dict(max_iter=60, learning_rate=0.05, max_leaf_nodes=4,
                                  l2_regularization=100.0, min_samples_leaf=200)),
                    ("small", dict(max_iter=150, learning_rate=0.05, max_leaf_nodes=8,
                                   l2_regularization=30.0, min_samples_leaf=100)),
                    ("mid", dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                 l2_regularization=5.0, min_samples_leaf=40)),
                    ("big(prod arm)", dict(max_iter=600, learning_rate=0.06,
                                           max_leaf_nodes=63, l2_regularization=1.0))):
        kw.update(early_stopping=True, validation_fraction=0.15,
                  n_iter_no_change=25, random_state=seed)
        m = HistGradientBoostingRegressor(**kw).fit(Xh, y, sample_weight=w)
        add(f"GBM {tag}", lambda p, m=m: m.predict(F.VIEWS["hand"](p["X"])),
            f"{m.n_iter_} trees")
    return out


def main():
    for run in ("run2", "run3"):
        diagnostics(run)

    for run in ("run2", "run3"):
        g0 = C.read_gen(run, 0)
        tr, hold = C.split_games(g0, 0.1, seed=7)
        evals = {"val0": C.pack(hold), "g1": C.pack(C.read_gen(run, 1))}
        c0 = C.const0(evals, TARGET)
        fl = {k: C.floor_of(v, TARGET, "run2", 12) for k, v in evals.items()}
        pool = C.pack(tr)
        P(f"\n### EXP4 {run}: train on all {len(tr)} gen-0 games, target={TARGET}", flush=True)
        P(f"    val0 = 50-game gen-0 holdout ({len(evals['val0']['X'])} uniq, "
          f"floor {fl['val0']:.4f}, const0 {c0['val0']:.4f})", flush=True)
        P(f"    g1   = all of gen 1 ({len(evals['g1']['X'])} uniq, "
          f"floor {fl['g1']:.4f}, const0 {c0['g1']:.4f})", flush=True)
        hdr = (f"    {'model':>20} {'note':>12} | {'train':>7} | {'val0':>7} "
               f"{'vs const0':>10} | {'g1':>7} {'vs const0':>10}")
        P(hdr); P("    " + "-" * (len(hdr) - 4), flush=True)
        allr = []
        for seed in SEEDS:
            allr += models(pool, evals, TARGET, seed)
        for name in [r["model"] for r in allr[:len(allr) // len(SEEDS)]]:
            sub = [r for r in allr if r["model"] == name]
            v = np.array([r["val0"] for r in sub]); g = np.array([r["g1"] for r in sub])
            t = np.array([r["train"] for r in sub])
            P(f"    {name:>20} {sub[0]['note']:>12} | {t.mean():>7.4f} | "
              f"{v.mean():>7.4f} {100*(v.mean()/c0['val0']-1):>+9.1f}% | "
              f"{g.mean():>7.4f} {100*(g.mean()/c0['g1']-1):>+9.1f}%", flush=True)
        C.save(f"exp4_gen0_{run}_{TARGET}.json",
               dict(run=run, target=TARGET, floors=fl, const0=c0, rows=allr))


if __name__ == "__main__":
    main()
