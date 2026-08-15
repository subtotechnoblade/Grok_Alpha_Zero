"""EXPERIMENT 1 - can the production ResNet reach the GBT's held-out value MSE?

Same file, same seed, same by-game split, same dedup, same held-out occurrence
weights as tmp/gbt_target.py. Only the TRAINING sample weights differ between arms.
Evaluation weighting is never changed - otherwise the number stops being comparable
to the GBT target.

  GBT [raw cells] : 0.6126   <- the bar, same information the ResNet gets
  GBT [hand+rel]  : 0.5681   <- stretch
  const-0         : 0.7691

arms:
  count   sample_weight = occurrence count (production today)
  uniform sample_weight = 1 for every unique position   <- experiment 1
"""
import os
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import h5py as h5
import tensorflow as tf

SRC = "Connect4/backup/Self_Play_Data.h5"
TEST_FRAC, SEED, BATCH = 0.1, 0, 1024
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 12


def load_split():
    """Byte-identical split to gbt_target.py: RandomState(0), by game, mirrors follow."""
    with h5.File(SRC, "r") as f:
        n = int(f["game_stats"][2])
        aug = ((len(f.keys()) - 1) // 3) // n
        rng = np.random.RandomState(SEED)
        test_ids = set(rng.permutation(n)[:int(TEST_FRAC * n)].tolist())
        out = {k: ([], [], []) for k in ("tr", "te")}
        for g in range(n):
            k = "te" if g in test_ids else "tr"
            for a in range(aug):
                i = g * aug + a
                out[k][0].append(f[f"boards_{i}"][:])
                out[k][1].append(f[f"policies_{i}"][:])
                out[k][2].append(f[f"values_{i}"][:][:, 0])
    return {k: tuple(np.concatenate(v) for v in out[k]) for k in out}


def dedup(X, P, y):
    order = {}
    for i, x in enumerate(X):
        order.setdefault(x.tobytes(), []).append(i)
    idx = np.array([v[0] for v in order.values()])
    cnt = np.array([len(v) for v in order.values()], dtype=np.float64)
    ym = np.array([y[v].mean() for v in order.values()])
    pm = np.array([P[v].mean(axis=0) for v in order.values()])
    return X[idx], pm, ym, cnt * (len(cnt) / cnt.sum()), cnt


def wmse(y, p, w):
    return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def build():
    from Connect4.Build_Model import build_model
    from Connect4.game import Connect4
    from Connect4.config import build_config, train_config, optimizer_config
    from Net.Alpha_Loss import Policy_Loss, Value_Loss
    from Net.Optimizers.adam import Adam
    from Net.Optimizers.nadam import Nadam
    from Net.Optimizers.muon import Muon

    g = Connect4()
    m = build_model(g.get_input_state().shape, g.policy_shape,
                    (build_config, train_config, optimizer_config))
    kw = dict(optimizer_config["kwargs"])
    kw["learning_rate"] = 3e-4
    opt = {"adam": Adam, "nadam": Nadam,
           "muon": Muon}[optimizer_config["optimizer"].lower()](**kw)
    m.compile(optimizer=opt, loss={"policy": Policy_Loss(dtype="float32"),
                                   "value": Value_Loss(dtype="float32")})
    return m


def main():
    t0 = time.time()
    d = load_split()
    trX, trP, trY, trW, _ = dedup(*d["tr"])
    teX, teP, teY, teW, _ = dedup(*d["te"])
    ply = (teX[..., -1] != 0).reshape(len(teX), -1).sum(1)
    base = wmse(teY, np.zeros(len(teY)), teW)
    print(f"train {len(trX)} unique | held-out {len(teX)} unique | "
          f"const-0 {base:.4f} | load {time.time()-t0:.0f}s")
    print(f"GBT bars -> raw cells 0.6126 | hand+rel 0.5681\n")

    Xtr = trX.astype(np.float32)
    Ptr = trP.astype(np.float32)
    Vtr = trY.astype(np.float32)[:, None]
    Xte = teX.astype(np.float32)

    arms = {"count (production)": trW.astype(np.float32),
            "uniform (EXP 1)": np.ones(len(trX), dtype=np.float32)}

    for name, w in arms.items():
        tf.keras.utils.set_random_seed(0)
        model = build()
        print(f"--- {name} ---  max w {w.max():.0f}")
        print(f"{'ep':>3} {'train_v':>8} {'HELD-OUT':>9} {'vs c0':>7} "
              f"{'ply0-9':>8} {'ply10-21':>9} {'ply22+':>8}")
        best = (1e9, -1)
        for ep in range(1, EPOCHS + 1):
            h = model.fit(Xtr, (Ptr, Vtr), sample_weight=(w, w), batch_size=BATCH,
                          epochs=1, verbose=0, shuffle=True)
            pv = model.predict(Xte, batch_size=4096, verbose=0)[1][:, 0].astype(np.float64)
            tot = wmse(teY, pv, teW)
            m1, m2, m3 = ply <= 9, (ply >= 10) & (ply <= 21), ply >= 22
            if tot < best[0]:
                best = (tot, ep)
            print(f"{ep:>3} {h.history['value_loss'][0]:>8.4f} {tot:>9.4f} "
                  f"{1-tot/base:>6.1%} {wmse(teY[m1],pv[m1],teW[m1]):>8.4f} "
                  f"{wmse(teY[m2],pv[m2],teW[m2]):>9.4f} "
                  f"{wmse(teY[m3],pv[m3],teW[m3]):>8.4f}", flush=True)
        print(f"    best held-out {best[0]:.4f} at epoch {best[1]}   "
              f"(GBT raw 0.6126)\n")


if __name__ == "__main__":
    main()
