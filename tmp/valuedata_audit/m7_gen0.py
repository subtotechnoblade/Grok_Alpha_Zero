"""M7: generation-0 specifics, because that is where the CURRENT run's fresh
data point lives (1000 games, q_weight=0, train 0.771->0.154 while val 0.808->0.959).

Three things the other scripts cannot answer for gen 0:
  (a) the ply-model floor is UNMEASURABLE past ply ~10 at gen 0 - nothing repeats.
      Report the coverage honestly instead of carrying a stale value forward.
  (b) how U/N (the only thing that separates the train floor from the val floor
      under dedup) scales with the number of games, so 500 -> 1000 can be read off.
  (c) the train-floor / val-floor RATIO at gen 0, which is what decides whether a
      6x train/val gap can be structural or must be generalisation failure.
"""
import numpy as np
import common as C

QW = {"run2": 0.2, "run3": 0.4}


def curve_with_coverage(z, ply, inv, ng):
    curve = np.full(42, np.nan)
    dfs = np.zeros(42)
    for p in range(42):
        m = ply == p
        if m.sum() < 2:
            continue
        si, sv = inv[m], z[m]
        n = np.bincount(si, minlength=ng).astype(np.float64)
        s = np.bincount(si, weights=sv, minlength=ng)
        s2 = np.bincount(si, weights=sv ** 2, minlength=ng)
        k = n >= 2
        df = (n[k] - 1).sum()
        dfs[p] = df
        if df > 0:
            curve[p] = (s2[k] - s[k] ** 2 / n[k]).sum() / df
    return curve, dfs


def gen0(run):
    games = C.load_games(run, 0, with_mirrors=True)
    keys, z, t, ply, gid = C.stack_samples(games)
    qw = QW[run]
    q = (t - (1 - qw) * z) / qw
    inv, counts, U, _ = C.group_index(keys)
    curve, dfs = curve_with_coverage(z, ply, inv, len(counts))
    cov = np.isfinite(curve[ply])
    print(f"\n===== {run} gen 0 (500 games, mirrored as training sees it) =====")
    print(f"  (a) ply-model floor coverage: only {cov.mean()*100:.1f}% of samples sit at a ply "
          f"with ANY repeated position. Measurable plies: "
          f"{[p for p in range(42) if np.isfinite(curve[p])]}")
    print(f"      sigma^2 where measurable: " +
          " ".join(f"p{p}={curve[p]:.3f}" for p in range(42) if np.isfinite(curve[p])))
    print(f"      -> the gen-0 floor past ply ~8 is NOT MEASURABLE from gen-0 data. "
          f"search-q upper bound on it: {float(((z-q)**2)[~cov].mean()):.4f} "
          f"(on the {(~cov).mean()*100:.0f}% of samples with no repeats)")
    print(f"      E[z^2] overall = {float((z**2).mean()):.4f};  search-q bound overall = "
          f"{float(((z-q)**2).mean()):.4f}")

    # (b) U/N vs number of games
    print(f"  (b) U/N vs games (drives the dedup floor; the CURRENT run uses 1000 games):")
    with C.open_gen(run, 0) as f:
        _, ngames, aug, _ = C.gen_meta(f)
    for frac in (0.125, 0.25, 0.5, 1.0):
        k = int(ngames * frac)
        sub = C.load_games(run, 0, max_games=k, with_mirrors=True)
        sk, sz, st, sp, sg = C.stack_samples(sub)
        si, sc, sU, _ = C.group_index(sk)
        print(f"      {k:>4} games: N={len(sz):>6}  unique={sU:>6}  U/N={sU/len(sz):.4f}  "
              f"mean occurrences/position={len(sz)/sU:.3f}")

    # (c) train / test floor ratio under the real 90/10 game split
    rng = np.random.RandomState(0)
    perm = rng.permutation(ngames)
    test_ids = set(perm[:int(0.1 * ngames)].tolist())
    is_test = np.array([g in test_ids for g in gid])
    print(f"  (c) train vs test side of the SAME gen-0 data (10% of games held out):")
    stats = {}
    for name, m in (("train", ~is_test), ("test", is_test)):
        ii, cc, uu, _ = C.group_index(keys[m])
        n = cc.astype(np.float64)
        # multiplicity: stored copies per distinct game in a group
        gg = gid[m]
        pair = np.unique(ii.astype(np.int64) * (gg.max() + 1) + gg)
        neff = np.bincount((pair // (gg.max() + 1)).astype(np.int64), minlength=len(n)).astype(float)
        mult = n / neff
        zz = z[m]
        zbar = np.bincount(ii, weights=zz, minlength=len(n)) / n
        const0 = float((n * zbar ** 2).sum() / len(zz))
        qe = np.bincount(ii, weights=((zz - q[m]) ** 2), minlength=len(n)) / n
        qfloor = float((mult * qe).sum() / len(zz))
        stats[name] = (len(zz), uu, uu / len(zz), qfloor, const0)
        print(f"      {name:>5}: N={len(zz):>6} unique={uu:>6} U/N={uu/len(zz):.4f} | "
              f"dedup-objective search-q floor bound={qfloor:.4f}  const-0={const0:.4f}")
    r = stats["test"][3] / stats["train"][3]
    ru = stats["test"][2] / stats["train"][2]
    print(f"      -> val-floor / train-floor = {r:.3f} (U/N ratio {ru:.3f}). "
          f"A PERFECT model would show val_loss ~= {r:.2f}x train_loss at gen 0.")


if __name__ == "__main__":
    for run in ("run2", "run3"):
        gen0(run)
