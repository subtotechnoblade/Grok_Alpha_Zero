"""M4: the floor OF THE OBJECTIVE KERAS ACTUALLY REPORTS, per split side, + overlap.

Replicates Dataloader.Create_Train_Test_Split exactly:
  window  = the last `num_previous_generations+1` gens, minus the oldest if gen>=2
  split   = by GAME, per generation, test_percent/train_percent with subtractive decay
  dedup   = per split side, byte-identical states merged to the mean target,
            weight = occurrence count normalised to mean 1.0 per side

Verified reduction convention (tiny keras probe, keras 3.7.0):
    reported loss = sum_i w_i * l_i / N_samples   and  mean(w)=1
so with w_j = n_j*U/N,   loss = (1/N_raw) * sum_over_unique n_j (pred_j - ybar_j)^2.

An ORACLE that predicts the true mu_j = E[y|s_j] therefore scores
    (1/N_raw) * sum_over_unique  (n_j / n_eff_j) * Var(y|s_j)
where n_eff_j = number of DISTINCT GAMES in the group (mirror copies of the same
game are not independent draws).  That is the number the reported val_loss must
be compared against.
"""
import json
import numpy as np
import common as C

QW = {"run2": 0.2, "run3": 0.4}
# current Connect4/config.py values
CFG = dict(num_previous_generations=4, train_percent=1.0, train_decay=0.2,
           test_percent=0.1, test_decay=0.02)


def window_for(gen, npg):
    """Mirrors Create_Train_Test_Split.__init__ file selection."""
    files = list(range(0, gen + 1))[-npg - 1:]
    if gen >= 2:
        files = files[1:]
    return files[::-1]  # newest first, the iteration order


def collect(run, gen, cfg=CFG, seed=0, max_gen_avail=None):
    """-> dict of arrays for train and test side (mirrors included, as training does)."""
    rng = np.random.RandomState(seed)
    win = window_for(gen, cfg["num_previous_generations"])
    tp, sp = cfg["train_percent"], cfg["test_percent"]
    side = {"train": [], "test": []}
    for g in win:
        games = C.load_games(run, g, with_mirrors=True)
        if not games:
            continue
        by_game = {}
        for rec in games:
            by_game.setdefault(rec["game"], []).append(rec)
        ng = len(by_game)
        perm = rng.permutation(ng)
        n_test = int(sp * ng)
        test_ids = set(perm[:n_test].tolist())
        rest = perm[n_test:]
        train_ids = set(rest[:int(tp * len(rest))].tolist())
        for gid, recs in by_game.items():
            if gid in test_ids:
                s = "test"
            elif gid in train_ids:
                s = "train"
            else:
                continue
            for rec in recs:
                b = rec["boards"]
                L = len(rec["target"])
                keys = [b[i].tobytes() for i in range(L)]
                side[s].append((keys, rec["z"].astype(np.float64),
                                rec["target"].astype(np.float64),
                                np.arange(L), np.full(L, g * 100000 + gid)))
        tp = max(0.0, tp - cfg["train_decay"])
        sp = max(0.0, sp - cfg["test_decay"])
    out = {}
    for s, lst in side.items():
        if not lst:
            out[s] = None
            continue
        out[s] = dict(
            keys=np.array([k for e in lst for k in e[0]], dtype=object),
            z=np.concatenate([e[1] for e in lst]),
            t=np.concatenate([e[2] for e in lst]),
            ply=np.concatenate([e[3] for e in lst]),
            gid=np.concatenate([e[4] for e in lst]))
    return out, win


def sigma_curve(y, ply, inv, ng):
    curve = np.full(42, np.nan)
    for p in range(42):
        m = ply == p
        if m.sum() < 2:
            continue
        si, sv = inv[m], y[m]
        n = np.bincount(si, minlength=ng).astype(np.float64)
        s = np.bincount(si, weights=sv, minlength=ng)
        s2 = np.bincount(si, weights=sv ** 2, minlength=ng)
        k = n >= 2
        df = (n[k] - 1).sum()
        if df > 0:
            curve[p] = (s2[k] - s[k] ** 2 / n[k]).sum() / df
    last = np.nan
    for p in range(42):
        if np.isnan(curve[p]):
            curve[p] = last
        else:
            last = curve[p]
    return curve


def dedup_stats(d, y, curve, qerr=None):
    """Return per-unique-position arrays for one split side."""
    inv, counts, U, _ = C.group_index(d["keys"])
    N = len(y)
    ng = len(counts)
    n_j = counts.astype(np.float64)
    ybar = np.bincount(inv, weights=y, minlength=ng) / n_j
    # n_eff = distinct games contributing to the group
    n_eff = np.array([len(np.unique(d["gid"][inv == j])) for j in range(ng)],
                     dtype=np.float64) if ng < 1 else None
    # (vectorised) distinct (group, game) pairs
    pair = inv.astype(np.int64) * (d["gid"].max() + 1) + d["gid"]
    uniq_pair = np.unique(pair)
    n_eff = np.bincount((uniq_pair // (d["gid"].max() + 1)).astype(np.int64),
                        minlength=ng).astype(np.float64)
    ply_j = np.bincount(inv, weights=d["ply"], minlength=ng) / n_j  # ply is constant in a group
    sig_j = curve[np.round(ply_j).astype(int)]
    res = dict(inv=inv, U=U, N=N, n_j=n_j, n_eff=n_eff, ybar=ybar, ply_j=ply_j,
               sig_j=sig_j, mult=n_j / n_eff)
    res["floor"] = float((res["mult"] * sig_j).sum() / N)
    res["const0"] = float((n_j * ybar ** 2).sum() / N)
    # empirical unbiased within-group variance where available (n_eff>=2), used as a
    # floor LOWER bound: singleton groups contribute >= 0
    s2 = np.bincount(inv, weights=y ** 2, minlength=ng)
    s1 = np.bincount(inv, weights=y, minlength=ng)
    ss = s2 - s1 ** 2 / n_j
    k = n_eff >= 2
    res["floor_lower_partial"] = float((ss[k] / (n_eff[k] - 1) * res["mult"][k]).sum() / N)
    if qerr is not None:
        qb = np.bincount(inv, weights=qerr, minlength=ng) / n_j
        res["floor_qbound"] = float((res["mult"] * qb).sum() / N)
    return res


def report(run, gen, target="z", cfg=CFG, seed=0, label=""):
    sides, win = collect(run, gen, cfg, seed)
    qw = QW[run]
    # sigma curve from the POOLED window (train+test), which is a property of the
    # data distribution not of the split
    allk = np.concatenate([sides[s]["keys"] for s in ("train", "test") if sides[s]])
    ally = np.concatenate([(sides[s]["z"] if target == "z" else sides[s]["t"])
                           for s in ("train", "test") if sides[s]])
    allp = np.concatenate([sides[s]["ply"] for s in ("train", "test") if sides[s]])
    inv_all, counts_all, U_all, _ = C.group_index(allk)
    curve = sigma_curve(ally, allp, inv_all, len(counts_all))

    print(f"\n########## {run} gen {gen} window={win} target={target} {label} ##########")
    print(f" pooled window: N={len(ally)} unique={U_all} (U/N={U_all/len(ally):.3f})")
    out = {}
    for s in ("train", "test"):
        if not sides[s]:
            continue
        d = sides[s]
        y = d["z"] if target == "z" else d["t"]
        q = (d["t"] - (1 - qw) * d["z"]) / qw
        qerr = (y - q) ** 2
        st = dedup_stats(d, y, curve, qerr)
        raw_floor = float(np.nanmean(curve[d["ply"]]))
        print(f"  {s:>5}: raw N={st['N']:>6}  unique={st['U']:>6}  U/N={st['U']/st['N']:.3f} "
              f"| DEDUP-OBJECTIVE  floor={st['floor']:.4f} "
              f"(lower-bd {st['floor_lower_partial']:.4f}, q-bound {st['floor_qbound']:.4f}) "
              f" const0={st['const0']:.4f} "
              f"| raw per-sample floor={raw_floor:.4f}")
        out[s] = st
    # ---- overlap
    if sides["train"] and sides["test"]:
        tr, te = sides["train"], sides["test"]
        ytr = tr["z"] if target == "z" else tr["t"]
        yte = te["z"] if target == "z" else te["t"]
        itr, ctr, Utr, dtr = C.group_index(tr["keys"])
        ite, cte, Ute, dte = C.group_index(te["keys"])
        tr_mean = np.bincount(itr, weights=ytr, minlength=Utr) / ctr
        te_mean = np.bincount(ite, weights=yte, minlength=Ute) / cte
        # map each test unique position to a train group if present
        tr_idx = np.full(Ute, -1, dtype=np.int64)
        for k, j in dte.items():
            h = dtr.get(k)
            if h is not None:
                tr_idx[j] = h
        seen = tr_idx >= 0
        w_te = cte.astype(np.float64)             # weight before mean-1 normalisation
        frac_w = float(w_te[seen].sum() / w_te.sum())
        frac_u = float(seen.mean())
        gap = np.abs(te_mean[seen] - tr_mean[tr_idx[seen]])
        sq = (te_mean[seen] - tr_mean[tr_idx[seen]]) ** 2
        # a net that has PERFECTLY memorised the train mean scores this on the seen part
        memo_loss_seen = float((w_te[seen] * sq).sum() / w_te.sum())
        st_te = out["test"]
        floor_seen = float((st_te["mult"][seen] * st_te["sig_j"][seen]).sum() / st_te["N"])
        floor_unseen = float((st_te["mult"][~seen] * st_te["sig_j"][~seen]).sum() / st_te["N"])
        const0_seen = float((cte[seen] * te_mean[seen] ** 2).sum() / st_te["N"])
        const0_unseen = float((cte[~seen] * te_mean[~seen] ** 2).sum() / st_te["N"])
        print(f"  OVERLAP: {frac_u*100:.1f}% of unique test positions "
              f"({frac_w*100:.1f}% of test WEIGHT) also occur in train")
        print(f"           |train_mean - test_mean| on those: mean={gap.mean():.4f} "
              f"median={np.median(gap):.4f} p90={np.quantile(gap,0.9):.4f}")
        print(f"           perfect-memorisation loss on the overlapped part = {memo_loss_seen:.4f} "
              f"(vs its oracle floor {floor_seen:.4f})")
        print(f"           floor split: overlapped {floor_seen:.4f} + unseen {floor_unseen:.4f} "
              f"= {floor_seen+floor_unseen:.4f} ; const0 split: {const0_seen:.4f} + {const0_unseen:.4f}")
        out["overlap"] = dict(frac_unique=frac_u, frac_weight=frac_w,
                              gap_mean=float(gap.mean()), gap_med=float(np.median(gap)),
                              memo_loss_seen=memo_loss_seen, floor_seen=floor_seen,
                              floor_unseen=floor_unseen)
    return out


if __name__ == "__main__":
    res = {}
    # (1) the CURRENT run's exact situation: generation 0, window = gen 0 only, pure z
    for run in ("run2", "run3"):
        res[f"{run}_g0_z"] = report(run, 0, target="z",
                                    label="<- matches the CURRENT run gen 0 (q_weight=0)")
    # (2) steady state: 4-generation window
    for run in ("run2", "run3"):
        for g in (5, 10, 15):
            try:
                res[f"{run}_g{g}_z"] = report(run, g, target="z")
                res[f"{run}_g{g}_t"] = report(run, g, target="t")
            except (OSError, KeyError) as e:
                print(f"  skip {run} {g}: {e}")
    with open("m4_split.json", "w") as f:
        json.dump({k: {kk: (vv if not isinstance(vv, dict) else
                            {a: b for a, b in vv.items() if np.isscalar(b)})
                       for kk, vv in v.items()} for k, v in res.items()}, f, indent=1, default=float)
