"""EXP 3 + 4 - the generation-0 question.

Verified from source, not assumed: Self_Play.py:44 passes `session=None` when
generation == 0, so gen 0 really is played with no network (uniform priors).

EXP 3 (does a bigger gen 0 give a better gen-1 model?)
  The gen-1 model is trained on the window ending at gen 0, which contains only
  gen 0 (Dataloader drops the oldest of 5, and gens -3..-1 do not exist), at
  train_percent 1.0 and test_percent 0.1.  So we reproduce exactly that: hold out
  10% of gen 0's games, train on N of the remaining 450, and score on
    val0 = the gen-0 holdout          (what production's val_loss would show)
    g1   = the whole of generation 1  (the distribution the gen-1 model deploys into)
  N in {56, 112, 225, 450} = an 8x span.

  A 900-game gen 0 is reachable without touching the live run: run2 gen 0 and
  run3 gen 0 are both no-network generations of the same game, i.e. draws from the
  SAME policy.  The cross-run control arm (train on run3 gen 0 only, score on run2
  gen 1) tests that interchangeability before the pooled arm is believed.
  Cross-run arms use target z, which is q_weight-independent (run2 0.2 vs run3 0.4).

EXP 4 (is gen-0 data learnable at all?)
  const-0 / train-mean / GBM / oracle against the measured floor, plus the
  singleton rate and the within-group variance of z by ply band.
"""
import sys
import time

import numpy as np

import common as C

TARGET = sys.argv[1] if len(sys.argv) > 1 else "z"
SEEDS = (0, 1, 2)
HOLD = 0.1                    # production test_percent for the first file


def oracle(pool, evals, target):
    y = pool[f"y_{target}"]
    table = dict(zip((b.tobytes() for b in pool["X"]), y))
    fb = float(np.sum(pool["w"] * y) / np.sum(pool["w"]))
    out = {}
    for nm, p in evals.items():
        ks = [b.tobytes() for b in p["X"]]
        hit = np.array([k in table for k in ks])
        pr = np.array([table.get(k, fb) for k in ks])
        out[f"{nm}_oracle"] = C.wmse(p[f"y_{target}"], pr, p["w"])
        out[f"{nm}_cov_w"] = float(p["w"][hit].sum() / p["w"].sum())
    return out


def learnability(run):
    """EXP 4 diagnostics on the raw gen-0 pool."""
    games = C.read_gen(run, 0)
    gs = C.D.group_stats(games, target="z")
    p = C.pack(games)
    ply = C.ply_of(p["X"])
    print(f"\n--- EXP4 learnability of {run} gen 0 ---")
    print(f"  {len(games)} games, {p['n_raw']} raw positions -> {len(p['X'])} unique")
    print(f"  singletons: {100*np.mean(p['count']==1):.1f}% of unique rows, "
          f"{100*p['w'][p['count']==1].sum()/p['w'].sum():.1f}% of loss weight")
    print(f"  count: mean {p['count'].mean():.2f}  p90 {np.percentile(p['count'],90):.0f}  "
          f"max {p['count'].max()}")
    print(f"  {'ply band':>10} {'uniq':>7} {'wt%':>6} {'mean cnt':>9} "
          f"{'within-grp var(z)':>18} {'floor=var/cnt':>14}")
    for lo, hi in ((0, 5), (6, 9), (10, 13), (14, 17), (18, 21), (22, 29), (30, 41)):
        m = (ply >= lo) & (ply <= hi)
        if not m.sum():
            continue
        v = gs["var"][m] if len(gs["var"]) == len(m) else None
        mm = (C.ply_of(gs["X"]) >= lo) & (C.ply_of(gs["X"]) <= hi)
        vv = gs["var"][mm]
        vv = vv[~np.isnan(vv)]
        var = float(vv.mean()) if len(vv) else float("nan")
        print(f"  {f'{lo}-{hi}':>10} {int(m.sum()):>7} "
              f"{100*p['w'][m].sum()/p['w'].sum():>5.1f}% {p['count'][m].mean():>9.2f} "
              f"{var:>18.4f} {var/max(p['count'][m].mean(),1):>14.4f}")
    return p


def main():
    t0 = time.time()
    rows = []
    for run in ("run2", "run3"):
        learnability(run)

    print(f"\n### EXP3 target={TARGET}: gen-0 size -> gen-1 quality")
    for run in ("run2", "run3"):
        g0 = C.read_gen(run, 0)
        g1 = C.read_gen(run, 1)
        tr_all, hold = C.split_games(g0, HOLD, seed=7)
        val0 = C.pack(hold)
        nxt1 = C.pack(g1)
        evals = {"val0": val0, "g1": nxt1}
        fl = {nm: C.floor_of(p, TARGET, "run2", 12) for nm, p in evals.items()}
        c0 = C.const0(evals, TARGET)
        print(f"\n== {run}: gen0 {len(g0)} games -> train pool {len(tr_all)}, "
              f"holdout {len(hold)}; gen1 {len(g1)} games ({len(nxt1['X'])} uniq)")
        print(f"   floors: val0 {fl['val0']:.4f}  g1 {fl['g1']:.4f}   "
              f"const0: val0 {c0['val0']:.4f}  g1 {c0['g1']:.4f}")
        hdr = (f"   {'arm':>16} {'games':>6} {'uniq':>7} {'seed':>4} {'model':>5} | "
               f"{'train':>7} {'val0':>7} {'v/fl':>6} | {'g1':>7} {'g1/fl':>6} "
               f"{'vs const0':>10} | {'g1 cov%':>7}")
        print(hdr); print("   " + "-" * (len(hdr) - 3))

        arms = [(f"{run} g0", tr_all, n) for n in (56, 112, 225, len(tr_all))]
        if run == "run2" and TARGET == "z":
            other = C.split_games(C.read_gen("run3", 0), HOLD, seed=7)[0]
            arms.append(("run3 g0 (control)", other, len(other)))
            arms.append(("run2+run3 g0", tr_all + other, len(tr_all) + len(other)))

        for arm, src, n in arms:
            for seed in SEEDS:
                rng = np.random.RandomState(2000 + seed)
                sel = rng.choice(len(src), size=min(n, len(src)), replace=False)
                pool = C.pack([src[i] for i in sel])
                orc = oracle(pool, evals, TARGET)
                for mn, fn in (("GBM", C.fit_gbm), ("MLP", C.fit_mlp)):
                    kw = dict(target=TARGET, seed=seed)
                    if mn == "MLP":
                        kw["val_for_es"] = val0
                    r = fn(pool, evals, **kw)
                    r.update(dict(run=run, arm=arm, model=mn, n_games=int(n), seed=seed,
                                  n_uniq=int(len(pool["X"])), target=TARGET))
                    r.update(orc)
                    rows.append(r)
                    print(f"   {arm:>16} {n:>6} {len(pool['X']):>7} {seed:>4} {mn:>5} | "
                          f"{r['train']:>7.4f} {r['val0']:>7.4f} "
                          f"{r['val0']/fl['val0']:>5.1f}x | {r['g1']:>7.4f} "
                          f"{r['g1']/fl['g1']:>5.1f}x {100*(r['g1']/c0['g1']-1):>+9.1f}% | "
                          f"{100*r['g1_cov_w']:>6.1f}", flush=True)
            sub = [r for r in rows if r["run"] == run and r["arm"] == arm
                   and r["n_games"] == n]
            for mn in ("GBM", "MLP"):
                v = np.array([r["g1"] for r in sub if r["model"] == mn])
                vv = np.array([r["val0"] for r in sub if r["model"] == mn])
                print(f"   {'':>16} {n:>6} {'':>7} {'MEAN':>4} {mn:>5} |         "
                      f"{vv.mean():>7.4f}+-{(vv.max()-vv.min())/2:<6.4f}   "
                      f"{v.mean():>7.4f}+-{(v.max()-v.min())/2:<6.4f}")

    C.save(f"exp3_gen0_{TARGET}.json", rows)
    print(f"\n{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
