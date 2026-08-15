"""M1/M2/M5: Bayes floor for predicting z, learnable variance, target consistency.

Three independent handles on the same quantity sigma^2 = E[Var(z|s)]:

  (a) REPEATS   pooled within-position variance of z over positions seen n>=2
                times. Unbiased for the repeat sub-population only.
  (b) SEARCH-Q  E[(z-q)^2] where q is the MCTS root value recovered from the
                stored target. q is a function of s (plus search noise), so
                E[(z-q)^2] >= E[Var(z|s)] ALWAYS: a certified UPPER BOUND on the
                floor that is valid for every sample, singletons included.
  (c) CONST     E[z^2] - what predicting 0 everywhere scores (no conditioning).

floor <= (b) and floor ~ (a); (c)-(floor) is the learnable variance.
"""
import sys, json
import numpy as np
import common as C

QW = {"run2": 0.2, "run3": 0.4}


def pooled_within_var(inv, vals, counts, mask):
    """Unbiased pooled within-group variance over groups with >=2 masked members."""
    sub_inv = inv[mask]
    sub_val = vals[mask]
    if len(sub_val) == 0:
        return np.nan, 0, 0, 0, np.nan
    ng = counts.shape[0]
    n = np.bincount(sub_inv, minlength=ng).astype(np.float64)
    s = np.bincount(sub_inv, weights=sub_val, minlength=ng)
    s2 = np.bincount(sub_inv, weights=sub_val ** 2, minlength=ng)
    keep = n >= 2
    ss_within = (s2[keep] - s[keep] ** 2 / n[keep]).sum()
    df = (n[keep] - 1).sum()
    n_members = n[keep].sum()
    # variance of the SAME sub-population, so the learnable fraction compares like
    # with like (repeated positions are not a random sample of all positions)
    memb_mask = keep[sub_inv]
    sub_all = sub_val[memb_mask]
    var_sub = sub_all.var() if len(sub_all) > 1 else np.nan
    return (ss_within / df if df > 0 else np.nan, int(df), int(keep.sum()),
            int(n_members), var_sub)


def analyse(run, gen, key_mode="state", verbose=True):
    games = C.load_games(run, gen)
    if not games:
        return None
    keys, z, t, ply, gid = C.stack_samples(games, key_mode=key_mode)
    qw = QW[run]
    q = (t - (1.0 - qw) * z) / qw
    inv, counts, U, _ = C.group_index(keys)
    n_per_sample = counts[inv]
    N = len(z)

    rows = []
    for name, lo, hi in C.BUCKETS + [("ALL", 0, 41)]:
        m = (ply >= lo) & (ply <= hi)
        if m.sum() == 0:
            continue
        zb, qb, tb = z[m], q[m], t[m]
        sig2, df, ngrp, nmemb, var_sub = pooled_within_var(inv, z, counts, m)
        # unique positions whose members fall in this bucket (ply is a function of
        # the board in connect4, so buckets do not split a group)
        u_bucket = len(np.unique(inv[m]))
        rows.append(dict(
            bucket=name, N=int(m.sum()), U=u_bucket,
            frac_n1=float((n_per_sample[m] == 1).mean()),
            mean_z=float(zb.mean()), var_z=float(zb.var()), Ez2=float((zb ** 2).mean()),
            sigma2_repeats=float(sig2), df=df, n_groups_ge2=ngrp, n_members=nmemb,
            var_z_repeatsub=float(var_sub),
            learnable_frac_repeats=float(1 - sig2 / var_sub) if var_sub and var_sub > 0 else np.nan,
            mse_q_z=float(((zb - qb) ** 2).mean()),
            mse_t_z=float(((zb - tb) ** 2).mean()),
            learnable_frac_q=float(1 - ((zb - qb) ** 2).mean() / zb.var()) if zb.var() > 0 else np.nan,
        ))
    if verbose:
        print(f"\n=== {run} gen {gen}  key={key_mode}  qw={qw}  "
              f"N={N} unique={U} ({U/N:.3f} U/N) ===")
        hdr = (f"{'bucket':>6} {'N':>6} {'uniq':>6} {'n=1':>5} | {'E[z]':>6} {'Var z':>6} "
               f"{'E[z^2]':>6} | {'sig2(rep)':>9} {'df':>6} {'VarZ|rep':>8} {'learn%':>6} | "
               f"{'MSE(q,z)':>8} {'learn%q':>7} {'MSE(t,z)':>8}")
        print(hdr)
        for r in rows:
            print(f"{r['bucket']:>6} {r['N']:>6} {r['U']:>6} {r['frac_n1']:>5.2f} | "
                  f"{r['mean_z']:>6.3f} {r['var_z']:>6.3f} {r['Ez2']:>6.3f} | "
                  f"{r['sigma2_repeats']:>9.4f} {r['df']:>6} {r['var_z_repeatsub']:>8.4f} "
                  f"{100*r['learnable_frac_repeats']:>6.1f} | "
                  f"{r['mse_q_z']:>8.4f} {100*r['learnable_frac_q']:>7.1f} {r['mse_t_z']:>8.4f}")
    return dict(run=run, gen=gen, key_mode=key_mode, N=N, U=U, rows=rows)


def count_bucket_breakdown(run, gen):
    """occurrence-count histogram, by weight, on the whole generation."""
    games = C.load_games(run, gen)
    keys, z, t, ply, gid = C.stack_samples(games)
    inv, counts, U, _ = C.group_index(keys)
    n = counts[inv]
    print(f"\n-- {run} gen {gen} occurrence-count structure (unmirrored, whole gen) --")
    for lo, hi, lbl in ((1, 1, "n=1"), (2, 9, "n=2-9"), (10, 10 ** 9, "n>=10")):
        m = (n >= lo) & (n <= hi)
        print(f"   {lbl:>7}: {m.sum():>6} samples ({m.mean()*100:>5.1f}%), "
              f"{len(np.unique(inv[m])):>6} unique positions, "
              f"E[z^2]={float((z[m]**2).mean()):.3f}, Var(z)={float(z[m].var()):.3f}")


if __name__ == "__main__":
    out = []
    for run, gens in (("run2", [0, 5, 10, 15]), ("run3", [0, 5, 10, 15])):
        for g in gens:
            r = analyse(run, g)
            if r:
                out.append(r)
    print("\n\n############ BOARD-ONLY KEY (drop the 2 history planes) ############")
    for run, gens in (("run2", [0, 10]), ("run3", [0, 10])):
        for g in gens:
            analyse(run, g, key_mode="board")
    for run, g in (("run2", 10), ("run3", 10)):
        count_bucket_breakdown(run, g)
    with open("m1_floor.json", "w") as f:
        json.dump(out, f, indent=1)
