"""EXP 2 - which axis carries the improvement: games PER generation, or MORE generations?

This is the experiment that reconciles the two earlier findings (subsampling games
inside the window helped; lengthening the window did not).  Both are "more training
samples", so if sample count were the whole story they would agree.  They differ in
WHERE the extra games come from, so the pool composition is varied with the total
held fixed.

Fixed evaluation for every arm (never in any training pool):
  nxt   = the whole of generation 13   -> the deployment distribution
  val12 = a 10%-by-game holdout of gen 12 -> an in-window-style holdout

Four families:
  A  CONSTANT TOTAL 450 games, spread over 1 / 2 / 3 / 5 / 9 generations ending at 12
  B  CONSTANT TOTAL 1800 games, spread over 4 / 6 / 9 / 12 generations ending at 12
  C  STALENESS: 450 games from a 2-generation block at distance 0 / 4 / 8 from gen 12
     (the block width, and so the games-per-generation, is identical - only the age
     of the policy that produced them changes)
  D  ADD GENERATIONS AT FULL SIZE: the owner's window-length sweep, replicated here
     so it sits on the same axes as the exp1 within-window curve
"""
import sys
import time

import numpy as np

import common as C

RUN = "run2"
GEN = 12
TARGET = sys.argv[1] if len(sys.argv) > 1 else "blend"
SEEDS = (0, 1, 2)
HOLD = 0.1


def build_pool(src, spec, seed):
    """spec = [(gen, n_games), ...]; draws n_games by GAME from each generation."""
    rng = np.random.RandomState(3000 + seed)
    out = []
    for g, n in spec:
        pool = src[g]
        k = min(n, len(pool))
        sel = rng.choice(len(pool), size=k, replace=False)
        out += [pool[i] for i in sel]
    return C.pack(out)


def main():
    t0 = time.time()
    # sources: gen 12 minus its holdout, every older gen in full
    src = {}
    tr12, hold12 = C.split_games(C.read_gen(RUN, GEN), HOLD, seed=7)
    src[GEN] = tr12
    for g in range(1, GEN):
        src[g] = C.read_gen(RUN, g)
    val12 = C.pack(hold12)
    nxt = C.pack(C.read_gen(RUN, GEN + 1))
    evals = {"val12": val12, "nxt": nxt}
    fl = {nm: C.floor_of(p, TARGET, RUN, GEN) for nm, p in evals.items()}
    c0 = C.const0(evals, TARGET)
    print(f"### EXP2 {RUN} target={TARGET}   gen12 train pool {len(tr12)} games, "
          f"holdout {len(hold12)}")
    print(f"  val12 {len(val12['X'])} uniq floor {fl['val12']:.4f} const0 {c0['val12']:.4f}")
    print(f"  nxt   {len(nxt['X'])} uniq floor {fl['nxt']:.4f} const0 {c0['nxt']:.4f}\n")

    def block(gens, per):
        return [(g, per) for g in gens]

    ARMS = []
    # A: constant total 450
    for k, per in ((1, 450), (2, 225), (3, 150), (5, 90), (9, 50)):
        ARMS.append((f"A total450 {k}gen x{per}", block(range(GEN, GEN - k, -1), per), 450))
    # B: constant total 1800
    for k, per in ((4, 450), (6, 300), (9, 200), (12, 150)):
        ARMS.append((f"B total1800 {k}gen x{per}", block(range(GEN, GEN - k, -1), per), 1800))
    # C: staleness at fixed width and fixed games-per-generation
    for dist in (0, 4, 8):
        gs = [GEN - dist, GEN - dist - 1]
        ARMS.append((f"C stale d={dist} gens{gs[0]},{gs[1]} x225", block(gs, 225), 450))
    # D: add generations at full size
    for k in (1, 2, 3, 4, 8, 12):
        ARMS.append((f"D fullgens {k}gen", block(range(GEN, GEN - k, -1), 500),
                     None))

    rows = []
    hdr = (f"{'arm':>26} {'games':>6} {'uniq':>7} {'sd':>3} | {'train':>7} "
           f"{'val12':>7} {'v/fl':>6} | {'nxt':>7} {'n/fl':>6} {'vs c0':>7}")
    print(hdr); print("-" * len(hdr))
    for name, spec, tot in ARMS:
        got = []
        for seed in SEEDS:
            pool = build_pool(src, spec, seed)
            ng = sum(min(n, len(src[g])) for g, n in spec)
            r = C.fit_gbm(pool, evals, target=TARGET, seed=seed)
            r.update(dict(arm=name, n_games=int(ng), seed=seed,
                          n_uniq=int(len(pool["X"])), n_raw=int(pool["n_raw"]),
                          spec=[[int(g), int(n)] for g, n in spec]))
            rows.append(r); got.append(r)
            print(f"{name:>26} {ng:>6} {len(pool['X']):>7} {seed:>3} | "
                  f"{r['train']:>7.4f} {r['val12']:>7.4f} "
                  f"{r['val12']/fl['val12']:>5.1f}x | {r['nxt']:>7.4f} "
                  f"{r['nxt']/fl['nxt']:>5.1f}x {100*(r['nxt']/c0['nxt']-1):>+6.1f}%",
                  flush=True)
        v = np.array([r["nxt"] for r in got]); vv = np.array([r["val12"] for r in got])
        print(f"{name:>26} {got[0]['n_games']:>6} {'':>7} {'AVG':>3} |         "
              f"{vv.mean():>7.4f}+-{(vv.max()-vv.min())/2:<6.4f}   "
              f"{v.mean():>7.4f}+-{(v.max()-v.min())/2:<6.4f}\n", flush=True)

    C.save(f"exp2_mix_{TARGET}.json",
           dict(run=RUN, gen=GEN, target=TARGET, floors=fl, const0=c0, rows=rows))
    print(f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
