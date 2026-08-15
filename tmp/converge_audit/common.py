"""Shared harness for the convergence audit.

Everything here is read-only with respect to the repo: the archives are opened
mode="r", nothing outside tmp/converge_audit/ is written.

Two model architectures exist in this repo and they are NOT interchangeable:

  "old"     git HEAD's Connect4/Build_Model.py.  Value head =
            Conv2D(8,3x3,bias) -> Reshape -> BN -> Dense(128) -> Dense(64).
            This is what produced tmp/run2_archive_gen0-19_OLDVALUEHEAD.
  "current" the working tree's Connect4/Build_Model.py.  Value head =
            Conv2D(16,1x1,no bias) -> BN -> Reshape -> Dense(32).
            This is what produced tmp/run3_archive_..._BOTTLENECK, and it is
            what tmp/weight_sweep.py / decay_sweep.py / window_sweep.py used.

Consequence: run2's archived model.weights.h5 CANNOT be loaded into the model the
sweeps trained.  Warm start from a real production checkpoint is only possible on
run3.  Verified in step0_verify.py.
"""
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.dirname(HERE)
ROOT = os.path.dirname(TMP)
for p in (os.path.join(TMP, "learnability"), TMP, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)

import json
import numpy as np
import tensorflow as tf

import data as D                                    # tmp/learnability/data.py
from valmetrics import ply_of, wmse, position_floor  # tmp/valmetrics.py

BATCH = 1024
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)


def cbuild(run, gen, n_prev=4, train_decay=0.1, test_decay=0.01,
           next_gen_holdout=True, seed=0, verbose=False):
    """D.build with an npz cache. Same semantics, keyed on every argument."""
    key = (f"{run}_g{gen}_np{n_prev}_td{train_decay}_sd{test_decay}"
           f"_nx{int(next_gen_holdout)}_s{seed}.npz")
    path = os.path.join(CACHE, key)
    keys = ("X", "policy", "y_blend", "y_z", "w", "count")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        out = {"run": run, "gen": gen, "q_weight": D.Q_WEIGHT[run]}
        for sp in ("train", "val", "nxt"):
            if f"{sp}_X" in z:
                out[sp] = {k: z[f"{sp}_{k}"] for k in keys}
                out[sp]["n_raw"] = int(z[f"{sp}_n_raw"])
        return out
    out = D.build(run, gen, n_prev=n_prev, train_decay=train_decay,
                  test_decay=test_decay, next_gen_holdout=next_gen_holdout,
                  seed=seed, verbose=verbose)
    flat = {}
    for sp in ("train", "val", "nxt"):
        if sp in out:
            for k in keys:
                flat[f"{sp}_{k}"] = out[sp][k]
            flat[f"{sp}_n_raw"] = np.int64(out[sp]["n_raw"])
    np.savez_compressed(path, **flat)
    return out


# --------------------------------------------------------------------- model
def _optimizer(lr, weight_decay=None):
    from Connect4.config import optimizer_config
    from Net.Optimizers.adam import Adam
    from Net.Optimizers.nadam import Nadam
    from Net.Optimizers.muon import Muon
    cls = {"adam": Adam, "nadam": Nadam,
           "muon": Muon}[optimizer_config["optimizer"].lower()]
    kw = dict(optimizer_config["kwargs"])
    kw["learning_rate"] = lr                 # a callable schedule cannot be passed
    if weight_decay is not None:
        kw["weight_decay"] = weight_decay
    return cls(**kw)                         # NO orthograd, NO grokfast


def build_and_compile(lr=3e-4, warm=None, weight_decay=None, arch="current"):
    """Build the real Connect4 model with the production optimizer.

    warm: path to a model.weights.h5 to load BEFORE compiling (production's
          warm start, pipeline.py:234).
    """
    from Connect4.game import Connect4
    from Connect4.config import build_config, train_config, optimizer_config
    from Net.Alpha_Loss import Policy_Loss, Value_Loss

    if arch == "current":
        from Connect4.Build_Model import build_model
    else:
        from old_build_model import build_model    # tmp/converge_audit/old_build_model.py

    g = Connect4()
    model = build_model(g.get_input_state().shape, g.policy_shape,
                        (build_config, train_config, optimizer_config))
    if warm is not None:
        model.load_weights(warm)
    model.compile(optimizer=_optimizer(lr, weight_decay),
                  loss={"policy": Policy_Loss(dtype="float32"),
                        "value": Value_Loss(dtype="float32")})
    return model


# ----------------------------------------------------------------- evaluation
def value_pred(model, Xf, batch=4096):
    return model.predict(Xf, batch_size=batch, verbose=0)[1][:, 0].astype(np.float64)


def prep_eval(d, splits, target, floors, tgt):
    """Cache float32 X / ply / floor once per split."""
    ev = {}
    for sp in splits:
        if sp not in d:
            continue
        s = d[sp]
        ply = ply_of(s["X"])
        ev[sp] = dict(Xf=s["X"].astype(np.float32), y=s[target], w=s["w"], ply=ply,
                      floor=position_floor(ply, s["count"], floors, tgt),
                      count=s["count"], X=s["X"])
    return ev


class EpochProbe(tf.keras.callbacks.Callback):
    """After every epoch, full-pass weighted value MSE on each named split.

    Keras' own `value_loss` in the fit log is a running average over the epoch's
    mini-batches (so it lags the end-of-epoch weights) and reduces as
    sum(w*l)/N rather than sum(w*l)/sum(w).  Both are recorded: `keras_*` from
    the logs dict, `wmse_*` from this full pass.
    """

    def __init__(self, ev, verbose=True, dense_until=10, every=5, total=None):
        super().__init__()
        self.ev = ev
        self.hist = []
        self.verbose = verbose
        self.dense_until = dense_until
        self.every = every
        self.total = total

    def on_epoch_end(self, epoch, logs=None):
        e1 = epoch + 1
        if not (e1 <= self.dense_until or e1 % self.every == 0 or e1 == self.total):
            return
        logs = logs or {}
        row = {"epoch": e1,
               "keras_train_value_loss": float(logs.get("value_loss", np.nan)),
               "keras_val_value_loss": float(logs.get("val_value_loss", np.nan))}
        for name, e in self.ev.items():
            row[f"wmse_{name}"] = wmse(e["y"], value_pred(self.model, e["Xf"]), e["w"])
        self.hist.append(row)
        if self.verbose:
            extra = "  ".join(f"{k[5:]} {row[k]:.4f}" for k in row if k.startswith("wmse_"))
            print(f"    epoch {row['epoch']:>3}  keras_train {row['keras_train_value_loss']:.4f}  "
                  f"{extra}", flush=True)


def floors_for(run, gen):
    p = os.path.join(TMP, "learnability", f"floorfull_{run}_g{gen}.json")
    if os.path.exists(p):
        return json.load(open(p))
    # fall back to the only measured file; the floor is a property of the game's
    # label noise per ply band, so reusing it across gens is an approximation.
    return json.load(open(os.path.join(TMP, "learnability", "floorfull_run2_g12.json")))


def arch_of(run):
    return "old" if run == "run2" else "current"


def weights_path(run, gen):
    return os.path.join(D.ROOTS[run], str(gen), "model.weights.h5")


def dump(name, obj):
    p = os.path.join(HERE, name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=1)
    print(f"[wrote] {p}")
