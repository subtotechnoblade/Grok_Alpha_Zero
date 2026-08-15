"""EXP 7 - WHERE does the missing coverage live, and does more data buy it back?

exp1 showed gen-13 weight coverage rising only 15.9% -> 20.5% while the training pool
grew 16x.  This splits that number by ply band and by pool size, with no model
involved, so the ceiling is visible independently of any learner:

  cov%      share of the band's loss weight whose exact position is in the pool
  drift     weighted MSE between the pool's deduped target and gen 13's deduped
            target ON THE COVERED POSITIONS - what a perfect memoriser would score
  unseen    weighted MSE of const-0 on the UNCOVERED positions - what any model is
            reduced to guessing there

If the uncovered weight sits in the opening (low ply), the pool and the deployment
distribution disagree about which openings get played, which is a policy-drift
problem that more games per generation cannot fix.  If it sits in the endgame, it is
genuine combinatorial novelty, which more games buys only very slowly.
"""
import sys

import numpy as np

import common as C

RUN = "run2"
GEN = 12
TARGET = sys.argv[1] if len(sys.argv) > 1 else "blend"
BANDS = [(0, 5), (6, 9), (10, 13), (14, 17), (18, 21), (22, 29), (30, 41)]


def equal_size_staleness():
    """Coverage vs pool age with the EVALUATION SET SIZE HELD FIXED.

    exp5 part 1 compared a 50-game gen-12 holdout against 500-game later
    generations; a 500-game set has ~10x more unique deep positions, so its
    coverage is mechanically lower and the two numbers are not comparable.
    Here every evaluation set is 100 games, so the only thing that changes across
    rows is WHICH generation produced them.
    """
    print("\n  --- equal-size control: pool = 350 games of gen 12, "
          "every eval set = 100 games ---")
    tr, hold = C.split_games(C.read_gen(RUN, GEN), 0.2, seed=7)   # 400 / 100
    pool = C.pack(tr[:350])
    tab = dict(zip((b.tobytes() for b in pool["X"]), pool[f"y_{TARGET}"]))
    print(f"  {'eval set':>22} {'uniq':>6} {'row cov%':>9} {'wt cov%':>8} "
          f"{'drift':>7} {'c0|cover':>9} {'c0|all':>7}")
    evsets = [("gen12 holdout (same gen)", hold[:100])]
    for g in range(GEN + 1, min(GEN + 7, 19)):
        gg = C.read_gen(RUN, g)
        if gg:
            evsets.append((f"gen{g} (+{g-GEN})", gg[:100]))
    for lab, gl in evsets:
        p = C.pack(gl)
        ks = [b.tobytes() for b in p["X"]]
        hit = np.array([k in tab for k in ks])
        pr = np.array([tab.get(k, 0.0) for k in ks])
        yy, ww = p[f"y_{TARGET}"], p["w"]
        dr = C.wmse(yy[hit], pr[hit], ww[hit]) if hit.any() else float("nan")
        c0h = C.wmse(yy[hit], np.zeros(hit.sum()), ww[hit]) if hit.any() else float("nan")
        print(f"  {lab:>22} {len(p['X']):>6} {100*np.mean(hit):>8.1f}% "
              f"{100*ww[hit].sum()/ww.sum():>7.1f}% {dr:>7.4f} {c0h:>9.4f} "
              f"{C.wmse(yy, np.zeros(len(yy)), ww):>7.4f}")


def main():
    W = C.D.raw_window(RUN, GEN, seed=0)
    games, nxt = W["train_games"], W["nxt"]
    ply = C.ply_of(nxt["X"])
    y, w = nxt[f"y_{TARGET}"], nxt["w"]
    print(f"### EXP7 {RUN}: coverage of gen {GEN+1} by ply band, target={TARGET}")
    print(f"    gen{GEN+1}: {len(nxt['X'])} unique positions from "
          f"{len(C.read_gen(RUN, GEN+1))} games\n")

    print("  cell = cov% then  drift/const0  BOTH restricted to the covered positions")
    print(f"  {'band':>7} {'uniq':>7} {'wt%':>6} {'const0':>7} " +
          " ".join(f"{'N='+str(n):>20}" for n in (80, 320, 1292)))
    tabs = {}
    for n in (80, 320, 1292):
        rng = np.random.RandomState(1000)
        sel = rng.choice(len(games), size=min(n, len(games)), replace=False)
        pool = C.pack([games[i] for i in sel])
        tab = dict(zip((b.tobytes() for b in pool["X"]), pool[f"y_{TARGET}"]))
        ks = [b.tobytes() for b in nxt["X"]]
        hit = np.array([k in tab for k in ks])
        pr = np.array([tab.get(k, 0.0) for k in ks])
        tabs[n] = (hit, pr)

    rows = []
    for lo, hi in BANDS:
        m = (ply >= lo) & (ply <= hi)
        if not m.sum():
            continue
        cells = []
        rec = {"band": f"{lo}-{hi}", "n": int(m.sum()),
               "wt": float(w[m].sum() / w.sum()),
               "const0": C.wmse(y[m], np.zeros(m.sum()), w[m])}
        for n in (80, 320, 1292):
            hit, pr = tabs[n]
            mh, mu = m & hit, m & ~hit
            cov = float(w[mh].sum() / w[m].sum())
            dr = C.wmse(y[mh], pr[mh], w[mh]) if mh.sum() else float("nan")
            c0h = C.wmse(y[mh], np.zeros(mh.sum()), w[mh]) if mh.sum() else float("nan")
            cells.append(f"{100*cov:>5.1f}% {dr:>6.3f}/{c0h:<6.3f}")
            rec[f"cov_{n}"] = cov
            rec[f"drift_{n}"] = dr
            rec[f"const0_covered_{n}"] = c0h
        rows.append(rec)
        print(f"  {rec['band']:>7} {rec['n']:>7} {100*rec['wt']:>5.1f}% "
              f"{rec['const0']:>7.4f} " + " ".join(f"{c:>20}" for c in cells))

    equal_size_staleness()

    hit, pr = tabs[1292]
    print(f"\n  overall: covered weight {100*w[hit].sum()/w.sum():.1f}%  "
          f"drift on covered {C.wmse(y[hit], pr[hit], w[hit]):.4f}  "
          f"const0 on uncovered {C.wmse(y[~hit], np.zeros((~hit).sum()), w[~hit]):.4f}")
    best = (w[hit].sum() * C.wmse(y[hit], pr[hit], w[hit]) +
            w[~hit].sum() * C.wmse(y[~hit], np.zeros((~hit).sum()), w[~hit])) / w.sum()
    print(f"  => a PERFECT memoriser of the 1292-game pool that predicts 0 elsewhere "
          f"scores {best:.4f} on gen {GEN+1}")
    print(f"     (floor {C.floor_of(nxt, TARGET, RUN, GEN):.4f}, "
          f"const-0 everywhere {C.wmse(y, np.zeros(len(y)), w):.4f})")

    # how fast is coverage growing?  fit cov = 1 - a*N^-b
    print("\n  coverage growth (overall weight):")
    for n in (80, 320, 1292):
        h, _ = tabs[n]
        print(f"    N={n:>5}  {100*w[h].sum()/w.sum():>5.2f}%")
    C.save(f"exp7_coverage_{TARGET}.json", rows)


if __name__ == "__main__":
    main()
