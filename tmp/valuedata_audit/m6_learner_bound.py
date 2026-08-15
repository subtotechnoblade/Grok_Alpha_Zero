"""M6: a RIGOROUS, search-free upper bound on the Bayes floor.

The repeat-based floor cannot be measured at gen 0 (almost nothing repeats past
ply 10) and the search-q bound is only near-rigorous (root Q shares the ply's
dirichlet noise with the move actually played, so it can peek slightly).

This is assumption-free: fit ANY function f(s) on a game-disjoint train split and
evaluate on held-out games.  The held-out MSE is an unbiased estimate of
E[(z-f(s))^2] >= E[Var(z|s)] = the Bayes floor.  So it is a valid CEILING on the
floor, and simultaneously an apples-to-apples "what a searchless learner gets on
this exact data" reference for the ResNet.

Reported both ways:
  RAW    per-sample MSE on held-out samples          (comparable to m3)
  DEDUP  the objective keras reports for val_loss    (comparable to val value_loss)
"""
import json
import numpy as np
import common as C
from sklearn.ensemble import HistGradientBoostingRegressor

QW = {"run2": 0.2, "run3": 0.4}


def tactical(board, mover):
    """board (6,7) int8 in absolute signs; returns a few cheap features."""
    mywin = oppwin = 0
    for c in range(7):
        for p, isme in ((mover, True), (-mover, False)):
            w = C._wins(board, c, p)
            if w:
                if isme:
                    mywin += 1
                else:
                    oppwin += 1
    legal = sum(1 for c in range(7) if np.count_nonzero(board[:, c]) < 6)
    return mywin, oppwin, legal


def featurise(games):
    X, Z, T, PLY, GID = [], [], [], [], []
    for g in games:
        b = g["boards"]
        for i in range(len(g["target"])):
            board = b[i, :, :, 3].astype(np.int8)
            mv = int(g["mover"][i])
            mw, ow, lg = tactical(board, mv)
            X.append(np.concatenate([(board * mv).ravel().astype(np.float32),
                                     [i, mw, ow, lg]]))
        Z.append(g["z"]); T.append(g["target"])
        PLY.append(np.arange(len(g["target"])))
        GID.append(np.full(len(g["target"]), g["game"]))
    return (np.array(X, dtype=np.float32), np.concatenate(Z).astype(np.float64),
            np.concatenate(T).astype(np.float64), np.concatenate(PLY),
            np.concatenate(GID))


def dedup_objective(keys, y, pred):
    """(1/N) * sum_over_unique n_j (mean_pred_j - ybar_j)^2 -- exactly keras' val_loss."""
    inv, counts, U, _ = C.group_index(keys)
    n = counts.astype(np.float64)
    ybar = np.bincount(inv, weights=y, minlength=len(n)) / n
    pbar = np.bincount(inv, weights=pred, minlength=len(n)) / n
    return float((n * (pbar - ybar) ** 2).sum() / len(y)), U, len(y)


def run(run_name, gen, seed=0):
    games = C.load_games(run_name, gen)
    if not games:
        return None
    X, z, t, ply, gid = featurise(games)
    keys, _, _, _, _ = C.stack_samples(games)
    qw = QW[run_name]
    q = (t - (1 - qw) * z) / qw

    ng = gid.max() + 1
    rng = np.random.RandomState(seed)
    perm = rng.permutation(ng)
    hold = set(perm[:int(0.2 * ng)].tolist())
    te = np.array([g in hold for g in gid])
    tr = ~te

    out = {"run": run_name, "gen": gen}
    for name, y in (("z", z), ("t", t)):
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                          max_leaf_nodes=63, l2_regularization=1.0,
                                          early_stopping=True, validation_fraction=0.1,
                                          random_state=seed)
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        raw = float(((p - y[te]) ** 2).mean())
        ded, U, N = dedup_objective(keys[te], y[te], p)
        const0 = float((y[te] ** 2).mean())
        _, dU, _ = C.group_index(keys[te])[1:3] if False else (None, None, None)
        ded0, _, _ = dedup_objective(keys[te], y[te], np.zeros_like(p))
        out[f"raw_{name}"] = raw
        out[f"dedup_{name}"] = ded
        out[f"raw_const0_{name}"] = const0
        out[f"dedup_const0_{name}"] = ded0
        # per bucket
        pb = {}
        for bn, lo, hi in C.BUCKETS:
            k = (ply[te] >= lo) & (ply[te] <= hi)
            if k.sum():
                pb[bn] = dict(N=int(k.sum()), gbm=float(((p[k] - y[te][k]) ** 2).mean()),
                              const0=float((y[te][k] ** 2).mean()))
        out[f"buckets_{name}"] = pb
    # search-q reference on the same held-out samples
    out["raw_qbound_z"] = float(((z[te] - q[te]) ** 2).mean())
    out["n_train"] = int(tr.sum()); out["n_test"] = int(te.sum())

    print(f"\n=== {run_name} gen {gen}: search-free learner (HistGBM, 42 board cells "
          f"+ ply + 3 tactical feats), trained on 80% of GAMES, scored on the other 20% ===")
    print(f"  held-out samples {out['n_test']}  (train {out['n_train']})")
    for name in ("z", "t"):
        print(f"  target={name}:  RAW held-out MSE = {out[f'raw_{name}']:.4f}   "
              f"(const-0 = {out[f'raw_const0_{name}']:.4f})   |   "
              f"DEDUP objective = {out[f'dedup_{name}']:.4f}  (const-0 = {out[f'dedup_const0_{name}']:.4f})")
        print("            by ply bucket: " + "  ".join(
            f"{b}: {v['gbm']:.4f} (const0 {v['const0']:.4f})" for b, v in out[f"buckets_{name}"].items()))
    print(f"  search-q reference on the same held-out samples: {out['raw_qbound_z']:.4f}")
    return out


if __name__ == "__main__":
    res = []
    for rn, gens in (("run2", [0, 5, 10, 15]), ("run3", [0, 5, 10, 15])):
        for g in gens:
            r = run(rn, g)
            if r:
                res.append(r)
    with open("m6_learner.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
