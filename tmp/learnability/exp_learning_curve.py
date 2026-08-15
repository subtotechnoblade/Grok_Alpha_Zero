"""Experiment B: is the value head DATA-limited?

Subsample the training pool BY GAME (10/25/50/100% of the window's games), re-run the
production dedup on each subsample, and train the same small learner. Subsampling by
game, not by deduped row, is deliberate: "we are data-starved" is a claim about how
many self-play GAMES exist, and halving the games halves both the number of distinct
positions and the number of votes averaged into each repeated position's label.

3 seeds per fraction, so the spread is visible and small gaps are not over-read.
"""
import json
import os
import sys

import numpy as np

import data
import features as F

FRACS = (0.10, 0.25, 0.50, 1.00)
SEEDS = (0, 1, 2)


def fit_mlp(pool, val, nxt, target, seed):
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    tf.keras.utils.set_random_seed(seed)
    view = "hand+rel"
    sc = StandardScaler().fit(F.VIEWS[view](pool["X"]))
    Xtr = sc.transform(F.VIEWS[view](pool["X"])).astype(np.float32)
    Xva = sc.transform(F.VIEWS[view](val["X"])).astype(np.float32)
    m = tf.keras.Sequential([tf.keras.layers.Input((Xtr.shape[1],)),
                             tf.keras.layers.Dense(32, activation="relu"),
                             tf.keras.layers.Dense(32, activation="relu"),
                             tf.keras.layers.Dense(1, activation="tanh")])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20,
                                          restore_best_weights=True)
    m.fit(Xtr, pool[f"y_{target}"].astype(np.float32), sample_weight=pool["w"],
          validation_data=(Xva, val[f"y_{target}"].astype(np.float32), val["w"]),
          epochs=200, batch_size=1024, verbose=0, callbacks=[es])
    out = {}
    for nm, p in (("val", val), ("nxt", nxt)):
        if p is None:
            continue
        pr = m.predict(sc.transform(F.VIEWS[view](p["X"])).astype(np.float32),
                       batch_size=4096, verbose=0).ravel()
        out[nm] = data.wmse(p[f"y_{target}"], pr, p["w"])
    out["train"] = data.wmse(pool[f"y_{target}"],
                             m.predict(Xtr, batch_size=4096, verbose=0).ravel(), pool["w"])
    return out


def fit_gbm(pool, val, nxt, target, seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    view = "hand+rel"
    X = F.VIEWS[view](pool["X"])
    m = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.06, max_leaf_nodes=63,
                                      l2_regularization=1.0, early_stopping=True,
                                      validation_fraction=0.15, n_iter_no_change=25,
                                      random_state=seed).fit(X, pool[f"y_{target}"],
                                                             sample_weight=pool["w"])
    out = {}
    for nm, p in (("val", val), ("nxt", nxt)):
        if p is None:
            continue
        out[nm] = data.wmse(p[f"y_{target}"], m.predict(F.VIEWS[view](p["X"])), p["w"])
    out["train"] = data.wmse(pool[f"y_{target}"], m.predict(X), pool["w"])
    return out


def main(run="run2", gen=12, target="blend"):
    print(f"### {run} g{gen} target={target}: learning curve by GAME count")
    W = data.raw_window(run, gen, seed=0)
    games, val, nxt = W["train_games"], W["val"], W.get("nxt")
    print(f"  {len(games)} training games in the window, "
          f"val {len(val['X'])} unique, nxt {len(nxt['X']) if nxt else 0} unique\n")

    rows = []
    for frac in FRACS:
        n = max(10, int(round(frac * len(games))))
        for seed in SEEDS:
            rng = np.random.RandomState(1000 + seed)
            sel = rng.choice(len(games), size=n, replace=False)
            pool = data.pack_games([games[i] for i in sel])
            for name, fn in (("MLP(32,32)", fit_mlp), ("HistGBM", fit_gbm)):
                r = fn(pool, val, nxt, target, seed)
                r.update(dict(model=name, frac=frac, n_games=n, seed=seed,
                              n_unique=int(len(pool["X"])), n_raw=int(pool["n_raw"])))
                rows.append(r)
                print(f"  {name:11s} frac={frac:.2f} games={n:5d} unique={len(pool['X']):6d} "
                      f"seed={seed}  train={r['train']:.4f} val={r['val']:.4f} "
                      f"nxt={r.get('nxt', float('nan')):.4f}", flush=True)

    print("\n  summary (mean +- half-range over 3 seeds)")
    for name in ("MLP(32,32)", "HistGBM"):
        for frac in FRACS:
            sub = [r for r in rows if r["model"] == name and r["frac"] == frac]
            for split in ("val", "nxt"):
                v = np.array([r[split] for r in sub if split in r])
                if not len(v):
                    continue
                print(f"    {name:11s} {split} frac={frac:.2f}: "
                      f"{v.mean():.4f} +- {(v.max()-v.min())/2:.4f}  "
                      f"(games={sub[0]['n_games']}, unique={sub[0]['n_unique']})")

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"curve_{run}_g{gen}_{target}.json")
    with open(path, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "run2",
         int(sys.argv[2]) if len(sys.argv) > 2 else 12,
         sys.argv[3] if len(sys.argv) > 3 else "blend")
