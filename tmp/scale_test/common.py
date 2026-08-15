"""Shared plumbing for the SCALE experiments.

Everything here reuses tmp/learnability/data.py (production split + dedup) and
tmp/valmetrics.py (ply, weighted MSE, per-position noise floor).  Nothing is
re-derived.

Key object: a POOL of games.  `data._read_gen(root, gen)` returns a list of
per-game dicts; every experiment in this folder builds a training pool by
selecting GAMES (never positions) and running `data.pack_games` on the result,
which applies the production dedup_average + mean-1 occurrence weights.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.dirname(HERE)
REPO = os.path.dirname(TMP)
sys.path.insert(0, os.path.join(TMP, "learnability"))
sys.path.insert(0, TMP)
sys.path.insert(0, REPO)

import data as D                      # noqa: E402
import features as F                  # noqa: E402
from valmetrics import ply_of, position_floor  # noqa: E402

wmse = D.wmse

_GEN_CACHE = {}


def read_gen(run, gen):
    """Cached list-of-games for one generation of one archive."""
    key = (run, gen)
    if key not in _GEN_CACHE:
        _GEN_CACHE[key] = D._read_gen(D.ROOTS[run], gen)
    return _GEN_CACHE[key]


def floors(run="run2", gen=12):
    return json.load(open(os.path.join(TMP, "learnability", f"floorfull_{run}_g{gen}.json")))


FL = None


def floor_of(split, target="blend", run="run2", gen=12):
    """Weighted mean irreducible floor sigma^2(ply)/count for a packed split."""
    global FL
    if FL is None:
        FL = floors(run, gen)
    ply = ply_of(split["X"])
    pf = position_floor(ply, split["count"], FL, target)
    return float(np.average(pf, weights=split["w"]))


def split_games(games, frac_holdout, seed):
    """Split a list of games into (train_games, holdout_games) BY GAME."""
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(games))
    n_h = int(round(frac_holdout * len(games)))
    return [games[i] for i in order[n_h:]], [games[i] for i in order[:n_h]]


def pack(games):
    return D.pack_games(games)


# ------------------------------------------------------------------ learners
def fit_gbm(pool, evals, target="blend", seed=0, view="hand+rel"):
    """Reference learner: HistGradientBoosting on the hand features.

    Returns {name: wmse} for every entry of `evals` plus 'train'.  Fast enough to
    run hundreds of arms; the headline points are re-confirmed with the real
    ResNet in exp_resnet.py.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = F.VIEWS[view](pool["X"])
    y = pool[f"y_{target}"]
    m = HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.06, max_leaf_nodes=63, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=25,
        random_state=seed).fit(X, y, sample_weight=pool["w"])
    out = {"train": wmse(y, m.predict(X), pool["w"]), "n_iter": int(m.n_iter_)}
    for nm, p in evals.items():
        if p is None:
            continue
        out[nm] = wmse(p[f"y_{target}"], m.predict(F.VIEWS[view](p["X"])), p["w"])
    return out


def fit_mlp(pool, evals, target="blend", seed=0, view="hand+rel", val_for_es=None):
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    tf.keras.utils.set_random_seed(seed)
    sc = StandardScaler().fit(F.VIEWS[view](pool["X"]))
    Xtr = sc.transform(F.VIEWS[view](pool["X"])).astype(np.float32)
    y = pool[f"y_{target}"].astype(np.float32)
    m = tf.keras.Sequential([tf.keras.layers.Input((Xtr.shape[1],)),
                             tf.keras.layers.Dense(32, activation="relu"),
                             tf.keras.layers.Dense(32, activation="relu"),
                             tf.keras.layers.Dense(1, activation="tanh")])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    cbs, vd = [], None
    if val_for_es is not None:
        vd = (sc.transform(F.VIEWS[view](val_for_es["X"])).astype(np.float32),
              val_for_es[f"y_{target}"].astype(np.float32), val_for_es["w"])
        cbs = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20,
                                                restore_best_weights=True)]
    m.fit(Xtr, y, sample_weight=pool["w"], validation_data=vd,
          epochs=200, batch_size=1024, verbose=0, callbacks=cbs)
    out = {"train": wmse(y, m.predict(Xtr, batch_size=4096, verbose=0).ravel(), pool["w"])}
    for nm, p in evals.items():
        if p is None:
            continue
        pr = m.predict(sc.transform(F.VIEWS[view](p["X"])).astype(np.float32),
                       batch_size=4096, verbose=0).ravel()
        out[nm] = wmse(p[f"y_{target}"], pr, p["w"])
    return out


def const0(evals, target="blend"):
    return {nm: wmse(p[f"y_{target}"], np.zeros(len(p["X"])), p["w"])
            for nm, p in evals.items() if p is not None}


def save(name, obj):
    path = os.path.join(HERE, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print(f"wrote {path}")
