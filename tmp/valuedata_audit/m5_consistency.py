"""M5: is the spread of z within a position consistent with sampling noise?

Within ONE generation the test has no content: the key is the full net input, and
a connect4 board cannot repeat inside a game, so every occurrence of a key comes
from a DIFFERENT game and is an i.i.d. draw by construction.  Any dispersion
statistic is then 1.0 by algebra.  (Stated, not assumed - the permutation control
below measures it.)

The test WITH content is across generations, because the training window pools 4
generations and dedup-average merges them into one target.  If mu(s) moves
between generations, the merged target is an average of different distributions
-> the label is stale, not just noisy.

Statistic, per key present in >=2 generations with >=10 total occurrences:
    chi2_j  = sum_g n_jg (zbar_jg - zbar_j)^2 / s2_j     df_j = (#gens - 1)
    s2_j    = the key's own unbiased pooled variance of z
Dispersion = sum_j chi2_j / sum_j df_j.  1.0 = homogeneous across generations.
CONTROL   = the same statistic after shuffling the generation labels within each
            key, which must come out at 1.0 if the machinery is right.
"""
import numpy as np
import common as C


def collect_window(run, gens):
    K, Z, P, G = [], [], [], []
    for g in gens:
        games = C.load_games(run, g)
        if not games:
            continue
        k, z, t, ply, gid = C.stack_samples(games)
        K.append(k); Z.append(z); P.append(ply)
        G.append(np.full(len(z), g, dtype=np.int32))
    return (np.concatenate(K), np.concatenate(Z), np.concatenate(P), np.concatenate(G))


def dispersion(inv, z, gsrc, ng, min_n=10, min_gens=2, rng=None):
    order = np.argsort(inv, kind="stable")
    inv_s, z_s, g_s = inv[order], z[order], gsrc[order]
    bounds = np.searchsorted(inv_s, np.arange(ng + 1))
    chi2 = df = 0.0
    used = 0
    per = []
    for j in range(ng):
        a, b = bounds[j], bounds[j + 1]
        n = b - a
        if n < min_n:
            continue
        zz, gg = z_s[a:b], g_s[a:b]
        if rng is not None:
            gg = gg[rng.permutation(n)]
        ug = np.unique(gg)
        if len(ug) < min_gens:
            continue
        zbar = zz.mean()
        s2 = zz.var(ddof=1)
        if s2 <= 1e-12:
            continue
        c = sum(len(zz[gg == u]) * (zz[gg == u].mean() - zbar) ** 2 for u in ug) / s2
        chi2 += c
        df += len(ug) - 1
        used += 1
        per.append((c, len(ug) - 1, n))
    return chi2, df, used, per


def run(run_name, gens):
    keys, z, ply, gsrc = collect_window(run_name, gens)
    inv, counts, U, _ = C.group_index(keys)
    ng = len(counts)
    c, d, u, per = dispersion(inv, z, gsrc, ng)
    rng = np.random.RandomState(0)
    c0, d0, u0, _ = dispersion(inv, z, gsrc, ng, rng=rng)
    print(f"\n=== {run_name} window {gens}: cross-generation homogeneity of z per position ===")
    print(f"  positions tested: {u}  (n>=10 and present in >=2 gens)   total chi2 df = {int(d)}")
    print(f"  DISPERSION (observed)          = {c/d:.3f}   [1.0 = same distribution every gen]")
    print(f"  DISPERSION (gen labels shuffled) = {c0/d0:.3f}   <- machinery control, must be ~1.0")
    # per-ply-bucket view
    ply_j = np.zeros(ng)
    np.add.at(ply_j, inv, ply)
    ply_j /= counts
    for bn, lo, hi in C.BUCKETS:
        sel = (ply_j >= lo) & (ply_j <= hi)
        idx = np.where(sel)[0]
        mask = np.isin(inv, idx)
        if mask.sum() == 0:
            continue
        inv2 = inv.copy()
        inv2[~mask] = -1
        cc, dd, uu, _ = dispersion(np.where(mask, inv, ng), z, gsrc, ng + 1)
        # simpler: restrict arrays
        cc, dd, uu, _ = dispersion(inv[mask], z[mask], gsrc[mask], ng)
        if dd > 0:
            print(f"    ply {bn:>6}: dispersion {cc/dd:.3f} over {uu} positions, df={int(dd)}")


if __name__ == "__main__":
    run("run2", [7, 8, 9, 10])
    run("run2", [12, 13, 14, 15])
    run("run3", [7, 8, 9, 10])
    run("run3", [12, 13, 14, 15])
    # ---- WITHIN-generation control on real data --------------------------------
    # split ONE generation's games into two pseudo-generations by game-id parity.
    # If the dispersion is ~1.0 here but 4-9 across real generations, the excess is
    # unambiguously policy drift between generations, not an inconsistent label.
    print("\n=== WITHIN-generation control: one gen split into 2 pseudo-gens by game parity ===")
    for rn, g in (("run2", 10), ("run2", 15), ("run3", 10), ("run3", 15)):
        games = C.load_games(rn, g)
        keys, z, t, ply, gid = C.stack_samples(games)
        inv, counts, U, _ = C.group_index(keys)
        pseudo = (gid % 2).astype(np.int32)
        c, d, u, _ = dispersion(inv, z, pseudo, len(counts))
        rng = np.random.RandomState(0)
        c0, d0, u0, _ = dispersion(inv, z, pseudo, len(counts), rng=rng)
        print(f"  {rn} gen {g}: dispersion {c/d:.3f} (shuffled control {c0/d0:.3f}) "
              f"over {u} positions, df={int(d)}")
