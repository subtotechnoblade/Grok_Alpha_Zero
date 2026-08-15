"""EXP 1 - learning curve by GAME, reported on BOTH splits.

Protocol
--------
Fixed evaluation targets, identical for every arm:
  val = production's in-window by-game holdout (10% -> 4% decayed) of the SAME
        window the training pool is drawn from.
  nxt = the whole of generation GEN+1.  Nothing is ever selected on it.

Training pools are drawn BY GAME from the production window (gens GEN-3..GEN with
the production train_percent decay 1.0/0.8/0.6/0.4), then production dedup_average
+ mean-1 occurrence weights are applied to the subsample.  Subsampling by game is
essential: positions inside one game are not independent, and halving the games
must also halve the number of votes averaged into each repeated position's label.

Also records, per size:
  * coverage  = share of nxt loss-weight whose exact position appears in train
  * nxt_seen / nxt_unseen = the loss split by that coverage
  * ORACLE    = perfect memorisation of the train pool (train group mean, else the
                train weighted mean).  This is the best any model could do given
                THIS much data, so it separates "not enough data" from "model".
"""
import sys
import time

import numpy as np

import common as C

RUN = sys.argv[1] if len(sys.argv) > 1 else "run2"
GEN = int(sys.argv[2]) if len(sys.argv) > 2 else 12
TARGET = sys.argv[3] if len(sys.argv) > 3 else "blend"
SEEDS = (0, 1, 2)


def oracle(pool, evals, target):
    y = pool[f"y_{target}"]
    table = dict(zip((b.tobytes() for b in pool["X"]), y))
    fallback = float(np.sum(pool["w"] * y) / np.sum(pool["w"]))
    out = {}
    for nm, p in evals.items():
        ks = [b.tobytes() for b in p["X"]]
        hit = np.array([k in table for k in ks])
        pr = np.array([table.get(k, fallback) for k in ks])
        yy, w = p[f"y_{target}"], p["w"]
        out[f"{nm}_oracle"] = C.wmse(yy, pr, w)
        out[f"{nm}_cov_w"] = float(w[hit].sum() / w.sum())
        out[f"{nm}_seen"] = C.wmse(yy[hit], pr[hit], w[hit]) if hit.any() else float("nan")
        out[f"{nm}_unseen"] = C.wmse(yy[~hit], pr[~hit], w[~hit]) if (~hit).any() else float("nan")
    return out


def main():
    t0 = time.time()
    W = C.D.raw_window(RUN, GEN, seed=0)
    games, val, nxt = W["train_games"], W["val"], W["nxt"]
    evals = {"val": val, "nxt": nxt}
    fl = {nm: C.floor_of(p, TARGET, RUN, GEN) for nm, p in evals.items()}
    c0 = C.const0(evals, TARGET)

    print(f"### EXP1 {RUN} g{GEN} target={TARGET}")
    print(f"  window gens {GEN-3}..{GEN}, {len(games)} train games "
          f"({sum(len(g['boards']) for g in games)} raw positions)")
    print(f"  val {len(val['X'])} uniq  floor={fl['val']:.4f}  const0={c0['val']:.4f}")
    print(f"  nxt {len(nxt['X'])} uniq  floor={fl['nxt']:.4f}  const0={c0['nxt']:.4f}\n")

    sizes = [80, 160, 320, 640, len(games)]
    rows = []
    hdr = (f"{'games':>6} {'uniq':>7} {'seed':>4} {'model':>8} | {'train':>7} "
           f"{'val':>7} {'v/fl':>6} | {'nxt':>7} {'n/fl':>6} | "
           f"{'cov%':>5} {'nxt_seen':>8} {'nxt_uns':>8} {'nxt_orac':>8}")
    print(hdr); print("-" * len(hdr))

    for n in sizes:
        for seed in SEEDS:
            rng = np.random.RandomState(1000 + seed)
            sel = rng.choice(len(games), size=min(n, len(games)), replace=False)
            pool = C.pack([games[i] for i in sel])
            orc = oracle(pool, evals, TARGET)
            for mname, fn in (("GBM", C.fit_gbm), ("MLP", C.fit_mlp)):
                kw = dict(target=TARGET, seed=seed)
                if mname == "MLP":
                    kw["val_for_es"] = val
                r = fn(pool, evals, **kw)
                r.update(dict(model=mname, n_games=int(n), seed=seed,
                              n_uniq=int(len(pool["X"])), n_raw=int(pool["n_raw"])))
                r.update(orc)
                rows.append(r)
                print(f"{n:>6} {len(pool['X']):>7} {seed:>4} {mname:>8} | "
                      f"{r['train']:>7.4f} {r['val']:>7.4f} {r['val']/fl['val']:>5.1f}x | "
                      f"{r['nxt']:>7.4f} {r['nxt']/fl['nxt']:>5.1f}x | "
                      f"{100*r['nxt_cov_w']:>4.1f} {r['nxt_seen']:>8.4f} "
                      f"{r['nxt_unseen']:>8.4f} {r['nxt_oracle']:>8.4f}", flush=True)

    print("\n  SUMMARY  mean +- half-range over seeds")
    print(f"  {'model':>6} {'games':>6} {'uniq':>7} | {'train':>15} {'val':>15} {'val/fl':>7} "
          f"| {'nxt':>15} {'nxt/fl':>7} | {'cov%':>5}")
    for mname in ("GBM", "MLP"):
        for n in sizes:
            sub = [r for r in rows if r["model"] == mname and r["n_games"] == n]
            def ms(k):
                v = np.array([r[k] for r in sub])
                return v.mean(), (v.max() - v.min()) / 2
            tm, th = ms("train"); vm, vh = ms("val"); nm_, nh = ms("nxt")
            cov = np.mean([r["nxt_cov_w"] for r in sub])
            print(f"  {mname:>6} {n:>6} {sub[0]['n_uniq']:>7} | {tm:>7.4f}+-{th:<7.4f} "
                  f"{vm:>7.4f}+-{vh:<7.4f} {vm/fl['val']:>6.1f}x | "
                  f"{nm_:>7.4f}+-{nh:<7.4f} {nm_/fl['nxt']:>6.1f}x | {100*cov:>4.1f}")

    C.save(f"exp1_{RUN}_g{GEN}_{TARGET}.json",
           dict(run=RUN, gen=GEN, target=TARGET, floors=fl, const0=c0,
                n_window_games=len(games), rows=rows))
    print(f"\n{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
