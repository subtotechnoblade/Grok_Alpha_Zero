"""Model zoo on the shared benchmark split. Writes results_<run>_g<gen>.json.

Protocol
--------
`val`  = production's own in-window 10%->4% BY-GAME holdout. Every model is allowed
         to pick hyperparameters / stop early on it, exactly as production does with
         ModelCheckpoint(monitor='val_loss', save_best_only=True). So val is an
         optimistic estimate for ALL models equally, and is the number directly
         comparable to the production val_loss the owner quotes.
`nxt`  = the whole NEXT generation. Nothing is ever selected on it. This is the
         clean number.

Metric = sum(w*(y-p)^2)/sum(w) with the production mean-1 occurrence weights, which
equals Keras' sample-weighted MSE over a full pass (mean(w) == 1 by construction).
"""
import json
import os
import sys
import time

import numpy as np

import data
import features as F

SEED = 0


# ------------------------------------------------------------------ helpers
def evaluate(name, pred_fn, d, splits, target, results, note=""):
    row = {"model": name, "target": target, "note": note}
    for s in splits:
        p = d[s]
        row[s] = data.wmse(p[f"y_{target}"], pred_fn(p), p["w"])
    results.append(row)
    print(f"  {name:38s} {target:5s} " +
          "  ".join(f"{s}={row[s]:.4f}" for s in splits) + (f"   [{note}]" if note else ""))
    return row


def wmean(y, w):
    return float(np.sum(w * y) / np.sum(w))


# ------------------------------------------------------------------ models
def run_constants(d, splits, target, results):
    tr = d["train"]
    y = tr[f"y_{target}"]
    evaluate("constant: predict 0", lambda p: np.zeros(len(p["X"])), d, splits, target, results)
    m = wmean(y, tr["w"])
    evaluate("constant: train weighted mean", lambda p, m=m: np.full(len(p["X"]), m),
             d, splits, target, results, note=f"mean={m:.4f}")


def run_ridge(d, splits, target, results):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    tr = d["train"]
    y = tr[f"y_{target}"]
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(y))
    cut = int(0.8 * len(idx))
    fit_i, sel_i = idx[:cut], idx[cut:]

    for view in ("raw", "rel", "hand", "hand+rel"):
        Xtr = F.VIEWS[view](tr["X"])
        sc = StandardScaler().fit(Xtr)
        Xs = sc.transform(Xtr)
        best, best_a = None, None
        for a in (1e-3, 1e-1, 1.0, 10.0, 100.0, 1000.0):
            m = Ridge(alpha=a).fit(Xs[fit_i], y[fit_i], sample_weight=tr["w"][fit_i])
            e = data.wmse(y[sel_i], m.predict(Xs[sel_i]), tr["w"][sel_i])
            if best is None or e < best:
                best, best_a = e, a
        model = Ridge(alpha=best_a).fit(Xs, y, sample_weight=tr["w"])
        cache = {}

        def pred(p, view=view, sc=sc, model=model, cache=cache):
            k = id(p)
            if k not in cache:
                cache[k] = model.predict(sc.transform(F.VIEWS[view](p["X"])))
            return cache[k]

        evaluate(f"ridge [{view}] ({Xtr.shape[1]}d)", pred, d, splits, target, results,
                 note=f"alpha={best_a:g}")


def _onehot_board(X):
    """mover-relative current board -> (N, 126) float32 one-hot for hamming via matmul."""
    X = np.asarray(X, dtype=np.int32)
    mover = -X[:, 0, 0, 0]
    b = (X[:, :, :, 3] * mover[:, None, None]).reshape(len(X), 42)
    oh = np.zeros((len(X), 42, 3), dtype=np.float32)
    np.put_along_axis(oh, (b + 1)[:, :, None], 1.0, axis=2)
    return oh.reshape(len(X), 126)


def run_knn(d, splits, target, results, ks=(1, 3, 10, 30), chunk=1024):
    tr = d["train"]
    y = tr[f"y_{target}"]
    cnt = tr["count"].astype(np.float64)
    A = _onehot_board(tr["X"])
    preds = {s: {k: np.zeros(len(d[s]["X"])) for k in ks} for s in splits}
    kmax = max(ks)
    for s in splits:
        B = _onehot_board(d[s]["X"])
        for i in range(0, len(B), chunk):
            sim = B[i:i + chunk] @ A.T                      # matches; hamming = 42 - sim
            part = np.argpartition(-sim, kmax - 1, axis=1)[:, :kmax]
            ordr = np.take_along_axis(sim, part, 1)
            srt = np.argsort(-ordr, axis=1)
            nb = np.take_along_axis(part, srt, 1)
            for k in ks:
                nk = nb[:, :k]
                wk = cnt[nk]
                preds[s][k][i:i + chunk] = (y[nk] * wk).sum(1) / wk.sum(1)
    for k in ks:
        row = {"model": f"kNN hamming k={k}", "target": target,
               "note": "occurrence-weighted mean of neighbours"}
        for s in splits:
            row[s] = data.wmse(d[s][f"y_{target}"], preds[s][k], d[s]["w"])
        results.append(row)
        print(f"  {'kNN hamming k=%d' % k:38s} {target:5s} " +
              "  ".join(f"{s}={row[s]:.4f}" for s in splits))


def run_trees(d, splits, target, results):
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    tr = d["train"]
    y = tr[f"y_{target}"]
    for view in ("hand", "hand+rel"):
        Xtr = F.VIEWS[view](tr["X"])
        m = HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.06, max_leaf_nodes=63, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=25,
            random_state=SEED).fit(Xtr, y, sample_weight=tr["w"])
        cache = {}

        def pred(p, view=view, m=m, cache=cache):
            k = id(p)
            if k not in cache:
                cache[k] = m.predict(F.VIEWS[view](p["X"]))
            return cache[k]
        evaluate(f"HistGradientBoosting [{view}]", pred, d, splits, target, results,
                 note=f"{m.n_iter_} trees")

    Xtr = F.VIEWS["hand"](tr["X"])
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=2,
                               random_state=SEED).fit(Xtr, y, sample_weight=tr["w"])
    cache = {}

    def predrf(p, cache=cache):
        k = id(p)
        if k not in cache:
            cache[k] = rf.predict(F.VIEWS["hand"](p["X"]))
        return cache[k]
    evaluate("RandomForest [hand]", predrf, d, splits, target, results)

    try:
        import xgboost as xgb
        Xtr = F.VIEWS["hand+rel"](tr["X"])
        bst = xgb.XGBRegressor(n_estimators=800, max_depth=7, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                               n_jobs=2, random_state=SEED, early_stopping_rounds=30,
                               eval_metric="rmse")
        rng = np.random.RandomState(SEED)
        idx = rng.permutation(len(y)); cut = int(0.85 * len(idx))
        fi, si = idx[:cut], idx[cut:]
        bst.fit(Xtr[fi], y[fi], sample_weight=tr["w"][fi],
                eval_set=[(Xtr[si], y[si])], sample_weight_eval_set=[tr["w"][si]], verbose=False)
        cache = {}

        def predx(p, cache=cache):
            k = id(p)
            if k not in cache:
                cache[k] = bst.predict(F.VIEWS["hand+rel"](p["X"]))
            return cache[k]
        evaluate("XGBoost [hand+rel]", predx, d, splits, target, results,
                 note=f"best_iter={bst.best_iteration}")
    except Exception as e:
        print(f"  [xgboost skipped: {type(e).__name__} {e}]")


def run_mlp(d, splits, target, results, view="hand+rel", hidden=(32, 32), tag=None, epochs=200):
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    tf.keras.utils.set_random_seed(SEED)
    tr, va = d["train"], d["val"]
    sc = StandardScaler().fit(F.VIEWS[view](tr["X"]))
    Xtr = sc.transform(F.VIEWS[view](tr["X"])).astype(np.float32)
    Xva = sc.transform(F.VIEWS[view](va["X"])).astype(np.float32)
    y = tr[f"y_{target}"].astype(np.float32)
    yv = va[f"y_{target}"].astype(np.float32)

    m = tf.keras.Sequential([tf.keras.layers.Input((Xtr.shape[1],))] +
                            [tf.keras.layers.Dense(h, activation="relu") for h in hidden] +
                            [tf.keras.layers.Dense(1, activation="tanh")])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20,
                                          restore_best_weights=True)
    m.fit(Xtr, y, sample_weight=tr["w"], validation_data=(Xva, yv, va["w"]),
          epochs=epochs, batch_size=1024, verbose=0, callbacks=[es])
    cache = {}

    def pred(p, cache=cache):
        k = id(p)
        if k not in cache:
            cache[k] = m.predict(sc.transform(F.VIEWS[view](p["X"])).astype(np.float32),
                                 batch_size=4096, verbose=0).ravel()
        return cache[k]
    name = tag or f"MLP {hidden} [{view}]"
    return evaluate(name, pred, d, splits, target, results,
                    note=f"{m.count_params()} params, stopped ep {es.stopped_epoch or epochs}")


def run_oracle(d, splits, target, results):
    """Perfect memorisation of TRAIN positions: look the state up, else train mean."""
    tr = d["train"]
    y = tr[f"y_{target}"]
    table = dict(zip((b.tobytes() for b in tr["X"]), y))
    fallback = wmean(y, tr["w"])
    cov = {}

    def pred(p):
        ks = [b.tobytes() for b in p["X"]]
        hit = np.array([k in table for k in ks])
        out = np.array([table.get(k, fallback) for k in ks])
        cov[id(p)] = (hit, out)
        return out
    row = evaluate("ORACLE lookup (train group means)", pred, d, splits, target, results)
    for s in splits:
        p = d[s]
        hit, out = cov[id(p)]
        wcov = float(p["w"][hit].sum() / p["w"].sum())
        seen = data.wmse(p[f"y_{target}"][hit], out[hit], p["w"][hit]) if hit.any() else float("nan")
        uns = data.wmse(p[f"y_{target}"][~hit], out[~hit], p["w"][~hit]) if (~hit).any() else float("nan")
        row[f"{s}_coverage_w"] = wcov
        row[f"{s}_seen_mse"] = seen
        row[f"{s}_unseen_mse"] = uns
        print(f"     -> {s}: {100*hit.mean():.1f}% of rows ({100*wcov:.1f}% of weight) also in train; "
              f"MSE on those = {seen:.4f}; on the rest = {uns:.4f}")


# ------------------------------------------------------------------ main
def main(run="run2", gen=12, targets=("blend", "z"), which=None, seed=None):
    global SEED
    if seed is not None:
        SEED = seed
    t0 = time.time()
    print(f"### {run} window gen {gen-3}-{gen} (production 4-gen window), "
          f"q_weight={data.Q_WEIGHT[run]}, split seed={SEED}")
    d = data.get(run, gen, seed=SEED)
    splits = [s for s in ("val", "nxt") if s in d]
    for s in ["train"] + splits:
        p = d[s]
        print(f"  {s}: {len(p['X'])} unique states, once-only {100*np.mean(p['count']==1):.1f}%")

    results = []
    stages = which or ["constants", "ridge", "knn", "trees", "mlp", "oracle"]
    for target in targets:
        print(f"\n-- target = {target} " + "-" * 50)
        if "constants" in stages: run_constants(d, splits, target, results)
        if "ridge" in stages: run_ridge(d, splits, target, results)
        if "knn" in stages: run_knn(d, splits, target, results)
        if "trees" in stages: run_trees(d, splits, target, results)
        if "mlp" in stages:
            run_mlp(d, splits, target, results, view="hand", hidden=(32, 32))
            run_mlp(d, splits, target, results, view="hand+rel", hidden=(32, 32))
            run_mlp(d, splits, target, results, view="rel", hidden=(64, 64))
        if "oracle" in stages: run_oracle(d, splits, target, results)

    suffix = "" if SEED == 0 else f"_s{SEED}"
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"results_{run}_g{gen}{suffix}.json")
    with open(out, "w") as f:
        json.dump({"run": run, "gen": gen, "q_weight": data.Q_WEIGHT[run],
                   "n_train": int(len(d["train"]["X"])),
                   "rows": results}, f, indent=1)
    print(f"\nwrote {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "run2",
         int(sys.argv[2]) if len(sys.argv) > 2 else 12,
         seed=int(sys.argv[3]) if len(sys.argv) > 3 else None)
