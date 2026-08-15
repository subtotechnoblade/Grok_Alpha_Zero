"""M0: verify the z reconstruction before anything is built on top of it.

Checks per generation:
  A. plane 0 encodes -next_player (mover)  -> at ply 0 mover must be -1 everywhere
  B. mover alternates every ply within a game
  C. counts of z[0] == +1 / -1 / 0 reproduce game_stats [p-1_wins, draws, p+1_wins]
  D. inferred q_weight: smallest qw putting 100% of q in [-1,1]
  E. with that qw, max|q| should be ~1.0 (a saturated search really does hit +-1)
"""
import sys, numpy as np
import common as C


def verify(run, gen):
    try:
        games = C.load_games(run, gen)
    except OSError as e:
        print(f"{run} gen {gen}: OPEN FAIL {e}")
        return None
    except KeyError as e:
        print(f"{run} gen {gen}: MISSING DATASET {e}")
        return None
    with C.open_gen(run, gen) as f:
        _, n_games, aug, gs = C.gen_meta(f)
    if not games:
        print(f"{run} gen {gen}: EMPTY (n_games={n_games}) - skipped")
        return None

    # A / B
    a_ok = all(int(g["mover"][0]) == -1 for g in games)
    b_ok = all(np.all(np.diff(g["mover"].astype(int)) == 0) is False for g in games)
    b_ok = all(np.all(g["mover"][1:].astype(int) == -g["mover"][:-1].astype(int)) for g in games)

    # C
    z0 = np.array([g["z"][0] for g in games])
    got = (int((z0 == 1).sum()), int((z0 == 0).sum()), int((z0 == -1).sum()))
    want = (int(gs[3]), int(gs[4]), int(gs[5]))
    c_ok = got == want

    # D / E
    qw, table = C.infer_q_weight(games)
    maxq = None
    if qw is not None and qw > 0:
        z = np.concatenate([g["z"] for g in games])
        t = np.concatenate([g["target"] for g in games])
        q = (t - (1 - qw) * z) / qw
        maxq = float(np.abs(q).max())

    lens = np.array([g["L"] for g in games])
    dec = np.array([g["decided"] for g in games])
    print(f"{run} g{gen:>2}: games={n_games} aug={aug} plies={lens.sum()} "
          f"avgL={lens.mean():.1f} decided={dec.mean():.3f} | "
          f"A(mover@0=-1)={a_ok} B(alternates)={b_ok} "
          f"C(z0 vs game_stats)={c_ok} got={got} want={want} | "
          f"qw={qw} max|q|={maxq if maxq is None else round(maxq,4)}")
    if not c_ok:
        print("   qw scan (qw, frac q in [-1,1], max|q|):",
              [(a, round(b, 4), round(c, 3)) for a, b, c in table])
    return dict(run=run, gen=gen, qw=qw, ok=(a_ok and b_ok and c_ok))


if __name__ == "__main__":
    res = []
    for run, gens in (("run2", range(20)), ("run3", range(16))):
        for g in gens:
            r = verify(run, g)
            if r:
                res.append(r)
    print("\nALL CHECKS PASS:", all(r["ok"] for r in res), f"({len(res)} generations)")
    print("q_weight by gen:")
    for run in ("run2", "run3"):
        print(" ", run, [(r["gen"], r["qw"]) for r in res if r["run"] == run])
