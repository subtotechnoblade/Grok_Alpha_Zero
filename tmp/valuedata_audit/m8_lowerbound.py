"""M8: a RIGOROUS LOWER bound on the dedup-objective floor at generation 0.

A floor cannot be lower-bounded from below on positions that never repeat - that
is a hard limit of the data, not a modelling choice.  But it CAN be on the slice
that does repeat, and every other position contributes >= 0.  So

    floor  >=  (1/N) * sum over unique positions with n_eff>=2 of  SS_j / (n_eff_j - 1)

with SS_j the sum of squares of z over the group's stored copies and n_eff_j the
number of DISTINCT GAMES in it (mirror copies of one game are one draw, not two).
Derivation: contribution of group j to the keras dedup loss for an oracle is
(n_j/N)*Var(zbar_j) = (n_j/N)*sigma_j^2/n_eff_j = (1/N)*mult_j*sigma_j^2, and
mult_j*sigma_j^2 = mult_j*(SS_j/mult_j)/(n_eff_j-1) = SS_j/(n_eff_j-1).

Printed cumulatively by ply cut-off, so the bound can be read at whatever depth
the coverage is still near-total.
"""
import numpy as np
import common as C


def bound(run, gen=0, test_frac=0.1, seed=0):
    games = C.load_games(run, gen, with_mirrors=True)
    keys, z, t, ply, gid = C.stack_samples(games)
    with C.open_gen(run, gen) as f:
        _, ngames, aug, _ = C.gen_meta(f)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(ngames)
    test_ids = set(perm[:int(test_frac * ngames)].tolist())
    is_test = np.array([g in test_ids for g in gid])

    print(f"\n===== {run} gen {gen}: rigorous LOWER bound on the dedup-objective floor =====")
    for side, m in (("train", ~is_test), ("test", is_test)):
        k, zz, pp, gg = keys[m], z[m], ply[m], gid[m]
        inv, counts, U, _ = C.group_index(k)
        ng = len(counts)
        n_j = counts.astype(np.float64)
        pair = np.unique(inv.astype(np.int64) * (gg.max() + 1) + gg)
        n_eff = np.bincount((pair // (gg.max() + 1)).astype(np.int64),
                            minlength=ng).astype(np.float64)
        s1 = np.bincount(inv, weights=zz, minlength=ng)
        s2 = np.bincount(inv, weights=zz ** 2, minlength=ng)
        SS = s2 - s1 ** 2 / n_j
        ply_j = np.bincount(inv, weights=pp, minlength=ng) / n_j
        contrib = np.where(n_eff >= 2, SS / np.maximum(n_eff - 1, 1), 0.0)
        N = len(zz)
        print(f"  {side}: N={N} unique={U}")
        print(f"      {'ply<=':>6} {'cover%':>7} {'LB(cum)':>8}   (cover% = share of this "
              f"side's samples at ply<=cut that sit in a repeated position)")
        for cut in (3, 5, 7, 9, 12, 41):
            sel = ply_j <= cut
            lb = float(contrib[sel].sum() / N)
            insel = pp <= cut
            rep = float((n_eff[inv][insel] >= 2).mean()) if insel.sum() else float("nan")
            print(f"      {cut:>6} {100*rep:>7.1f} {lb:>8.4f}")


if __name__ == "__main__":
    for r in ("run2", "run3"):
        bound(r, 0)
