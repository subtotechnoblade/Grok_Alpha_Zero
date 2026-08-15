"""M2: does sigma^2 measured on REPEATED positions extrapolate to SINGLETONS?

The floor estimate from repeats only sees positions with n>=2.  ~92% of the test
pool is n=1, so the whole audit hinges on whether Var(z|s) is the same for rare
and popular positions AT THE SAME PLY.

Test: stratify sigma^2 by (exact ply) x (occurrence-count band).  If sigma^2 is
flat across count bands within a ply, extrapolating to n=1 is safe; if it trends,
the trend gives the extrapolation.

Also emits the exact-ply sigma^2 curve used by m3/m4 as the floor model.
"""
import json
import numpy as np
import common as C

QW = {"run2": 0.2, "run3": 0.4}
CBANDS = [("2-3", 2, 3), ("4-9", 4, 9), ("10-49", 10, 49), ("50+", 50, 10 ** 9)]


def within_var(sub_inv, sub_val, ng):
    n = np.bincount(sub_inv, minlength=ng).astype(np.float64)
    s = np.bincount(sub_inv, weights=sub_val, minlength=ng)
    s2 = np.bincount(sub_inv, weights=sub_val ** 2, minlength=ng)
    keep = n >= 2
    ss = (s2[keep] - s[keep] ** 2 / n[keep]).sum()
    df = (n[keep] - 1).sum()
    return (ss / df if df > 0 else np.nan), int(df), int(keep.sum())


def run(run_name, gen, pool_gens=None):
    gens = pool_gens or [gen]
    allk, allz, allply, allgen = [], [], [], []
    for g in gens:
        games = C.load_games(run_name, g)
        if not games:
            continue
        k, z, t, ply, gid = C.stack_samples(games)
        allk.append(k); allz.append(z); allply.append(ply)
        allgen.append(np.full(len(z), g, dtype=np.int32))
    keys = np.concatenate(allk); z = np.concatenate(allz)
    ply = np.concatenate(allply); gsrc = np.concatenate(allgen)
    inv, counts, U, _ = C.group_index(keys)
    ng = len(counts)
    npos = counts[inv]

    tag = f"{run_name} gen {gens if pool_gens else gen}"
    print(f"\n===== sigma^2 by ply-bucket x occurrence-count band :: {tag} "
          f"(N={len(z)}, unique={U}) =====")
    print(f"{'bucket':>7} | " + " | ".join(f"{lbl:>18}" for lbl, _, _ in CBANDS))
    print(f"{'':>7} | " + " | ".join(f"{'sig2   df  grps':>18}" for _ in CBANDS))
    for bname, lo, hi in C.BUCKETS:
        cells = []
        for lbl, clo, chi in CBANDS:
            m = (ply >= lo) & (ply <= hi) & (npos >= clo) & (npos <= chi)
            if m.sum() < 2:
                cells.append(f"{'-':>18}")
                continue
            s2, df, grp = within_var(inv[m], z[m], ng)
            cells.append(f"{s2:>7.4f} {df:>5} {grp:>4}")
        print(f"{bname:>7} | " + " | ".join(cells))

    # exact-ply sigma^2 curve (all n>=2 groups at that ply) + singleton share
    print(f"\n{'ply':>4} {'N':>6} {'n=1%':>5} {'sig2':>7} {'df':>6} {'Var z':>7} {'E[z^2]':>7}")
    curve = {}
    for p in range(0, 42):
        m = ply == p
        if m.sum() == 0:
            continue
        s2, df, grp = within_var(inv[m], z[m], ng)
        curve[p] = dict(sigma2=None if np.isnan(s2) else float(s2), df=df,
                        N=int(m.sum()), var=float(z[m].var()))
        if p % 3 == 0 or p < 6:
            print(f"{p:>4} {m.sum():>6} {100*(npos[m]==1).mean():>5.1f} "
                  f"{s2:>7.4f} {df:>6} {z[m].var():>7.4f} {(z[m]**2).mean():>7.4f}")
    return curve


if __name__ == "__main__":
    out = {}
    for rn, g in (("run2", 10), ("run2", 15), ("run3", 10), ("run3", 5)):
        out[f"{rn}_g{g}"] = run(rn, g)
    # pooled 4-gen window = what training actually sees
    out["run2_w7-10"] = run("run2", None, pool_gens=[7, 8, 9, 10])
    out["run3_w7-10"] = run("run3", None, pool_gens=[7, 8, 9, 10])
    with open("m2_sigma_curve.json", "w") as f:
        json.dump(out, f, indent=1)
