"""EXP 5 - is the nxt error a DATA-VOLUME problem or a DISTRIBUTION-SHIFT problem?

The exp1 coverage column shows only ~17% of gen-13 loss weight sits on positions
that appear anywhere in the gens 9-12 training pool, and that share barely moves as
the pool grows.  That points at shift, not volume.  Two direct tests:

PART 1  overlap and target drift, gen G train pool vs gen G+k, for k = 0..6.
        - weight coverage: share of the later generation's loss weight whose exact
          position is present in the pool
        - target drift: on the covered positions, the weighted MSE between the pool's
          own deduped target and the later generation's deduped target.  This is an
          upper bound on how well a PERFECT memoriser of the pool could score there,
          and it is pure shift - no model, no capacity, no optimisation.

PART 2  the volume-vs-shift race.  Same learner, same evaluation set (a by-game
        holdout of gen 13), two training pools:
          IN-DISTRIBUTION, SMALL : 400 games of gen 13 itself
          OFF-DISTRIBUTION, BIG  : the full gens 9-12 window (~1260 games), and the
                                   gens 1-12 pool (~5400 games)
        If 400 in-distribution games beat 5400 off-distribution ones, more self-play
        games cannot be the fix - the pool would have to be fresher, not bigger.
"""
import sys
import time

import numpy as np

import common as C

RUN = "run2"
TARGET = sys.argv[1] if len(sys.argv) > 1 else "blend"
SEEDS = (0, 1, 2)


def overlap(pool, later, target):
    tab = dict(zip((b.tobytes() for b in pool["X"]), pool[f"y_{target}"]))
    ks = [b.tobytes() for b in later["X"]]
    hit = np.array([k in tab for k in ks])
    w, y = later["w"], later[f"y_{target}"]
    cov = float(w[hit].sum() / w.sum())
    if hit.any():
        pr = np.array([tab[k] for k, h in zip(ks, hit) if h])
        drift = C.wmse(y[hit], pr, w[hit])
    else:
        drift = float("nan")
    return cov, drift, float(np.mean(hit))


def part1():
    print("### EXP5 PART 1 - overlap and target drift as the pool ages")
    G = 12
    tr, hold = C.split_games(C.read_gen(RUN, G), 0.1, seed=7)
    # the production window pool (gens 9-12, decayed) for reference
    W = C.D.raw_window(RUN, G, seed=0)
    pools = {"gen12 only (450 g)": C.pack(tr),
             "window 9-12 (1260 g)": C.pack(W["train_games"])}
    print(f"\n  {'pool':>22} {'target':>13} {'row%':>6} {'wt cov%':>8} "
          f"{'drift(MSE)':>11} {'floor':>7} {'const0':>7}")
    rows = []
    for pname, pool in pools.items():
        for k in range(0, 7):
            later_games = C.read_gen(RUN, G + k)
            if not later_games:
                continue
            if k == 0:
                later = C.pack(hold)          # the honest in-generation holdout
                lab = "gen12 holdout"
            else:
                later = C.pack(later_games)
                lab = f"gen{G+k}"
            cov, drift, rowfrac = overlap(pool, later, TARGET)
            flr = C.floor_of(later, TARGET, RUN, G)
            c0 = C.wmse(later[f"y_{TARGET}"], np.zeros(len(later["X"])), later["w"])
            rows.append(dict(pool=pname, k=k, label=lab, cov_w=cov, row_frac=rowfrac,
                             drift=drift, floor=flr, const0=c0))
            print(f"  {pname:>22} {lab:>13} {100*rowfrac:>5.1f}% {100*cov:>7.1f}% "
                  f"{drift:>11.4f} {flr:>7.4f} {c0:>7.4f}")
    return rows


def part2():
    print("\n### EXP5 PART 2 - 400 fresh games vs 1260 and 5400 stale ones")
    g13 = C.read_gen(RUN, 13)
    tr13, hold13 = C.split_games(g13, 0.2, seed=11)      # 400 train / 100 eval
    ev = {"g13_hold": C.pack(hold13)}
    fl = C.floor_of(ev["g13_hold"], TARGET, RUN, 12)
    c0 = C.const0(ev, TARGET)["g13_hold"]
    print(f"  eval = gen13 by-game holdout, {len(hold13)} games, "
          f"{len(ev['g13_hold']['X'])} uniq, floor {fl:.4f}, const0 {c0:.4f}\n")

    W = C.D.raw_window(RUN, 12, seed=0)
    old_pool = W["train_games"]
    big = []
    for g in range(1, 13):
        big += C.read_gen(RUN, g)

    arms = [("IN-DIST gen13 x100", tr13, 100),
            ("IN-DIST gen13 x200", tr13, 200),
            ("IN-DIST gen13 x400", tr13, 400),
            ("STALE window 9-12", old_pool, len(old_pool)),
            ("STALE gens 1-12", big, len(big))]
    hdr = (f"  {'arm':>20} {'games':>6} {'uniq':>7} {'sd':>3} | {'train':>7} "
           f"{'g13 hold':>9} {'/floor':>7} {'vs const0':>10} {'cov%':>6}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    rows = []
    for name, src, n in arms:
        got = []
        for seed in SEEDS:
            rng = np.random.RandomState(4000 + seed)
            sel = rng.choice(len(src), size=min(n, len(src)), replace=False)
            pool = C.pack([src[i] for i in sel])
            cov, _, _ = overlap(pool, ev["g13_hold"], TARGET)
            r = C.fit_gbm(pool, ev, target=TARGET, seed=seed)
            r.update(dict(arm=name, n_games=int(min(n, len(src))), seed=seed,
                          n_uniq=int(len(pool["X"])), cov_w=cov))
            rows.append(r); got.append(r)
            print(f"  {name:>20} {r['n_games']:>6} {r['n_uniq']:>7} {seed:>3} | "
                  f"{r['train']:>7.4f} {r['g13_hold']:>9.4f} {r['g13_hold']/fl:>6.1f}x "
                  f"{100*(r['g13_hold']/c0-1):>+9.1f}% {100*cov:>5.1f}", flush=True)
        v = np.array([r["g13_hold"] for r in got])
        print(f"  {name:>20} {got[0]['n_games']:>6} {'':>7} {'AVG':>3} |           "
              f"{v.mean():>9.4f}+-{(v.max()-v.min())/2:.4f}\n", flush=True)
    return rows, dict(floor=fl, const0=c0)


if __name__ == "__main__":
    t0 = time.time()
    r1 = part1()
    r2, meta = part2()
    C.save(f"exp5_shift_{TARGET}.json", dict(part1=r1, part2=r2, part2_meta=meta))
    print(f"{time.time()-t0:.0f}s")
