"""EXPERIMENT 2 - optimizer / weight decay, on the exact split from gbt_target.py.

The value head is already Conv2D(2,(1,1)) in Connect4/Build_Model.py, so exp1's
0.6802 already included that change. The only remaining variable is the optimizer.

2x2 so the optimizer and the decay are separable rather than confounded:
  muon   wd 0.004 (default)   <- exp1 baseline, re-run here for a like-for-like row
  muon   wd 0.1
  adamw  wd 0.004
  adamw  wd 0.1               <- the requested arm

  GBT [raw cells] 0.6126 | GBT [hand+rel] 0.5681 | const-0 0.7691
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

SRC = os.environ.get("BENCH_H5", "Connect4/Grok_Zero_Train/0/Self_Play_Data.h5")
TEST_FRAC, SEED, BATCH = 0.1, 0, 1024
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def load_split():
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
    return X[idx], pm, ym, cnt * (len(cnt) / cnt.sum())


def wmse(y, p, w):
    return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def build(kind, wd, ortho=False):
    from Connect4.Build_Model import build_model
    from Connect4.game import Connect4
    from Connect4.config import build_config, train_config, optimizer_config
    from Net.Alpha_Loss import Policy_Loss, Value_Loss
    from Net.Optimizers.adam import Adam
    from Net.Optimizers.muon import Muon
    from Net.Optimizers.ortho_grad import Orthograd

    g = Connect4()
    m = build_model(g.get_input_state().shape, g.policy_shape,
                    (build_config, train_config, optimizer_config))
    if kind == "muon":
        kw = dict(optimizer_config["kwargs"])
        kw.update(learning_rate=3e-4, weight_decay=wd)
        opt = Muon(**kw)
    else:
        opt = Adam(learning_rate=3e-4, weight_decay=wd, beta_1=0.9, beta_2=0.995)
    if ortho:
        opt = Orthograd(opt, exclude_layers=optimizer_config.get("orthograd_exclude", ()))
    m.compile(optimizer=opt, loss={"policy": Policy_Loss(dtype="float32"),
                                   "value": Value_Loss(dtype="float32")})
    return m


def main():
    d = load_split()
    trX, trP, trY, trW = dedup(*d["tr"])
    teX, teP, teY, teW = dedup(*d["te"])
    ply = (teX[..., -1] != 0).reshape(len(teX), -1).sum(1)
    base = wmse(teY, np.zeros(len(teY)), teW)
    print(f"train {len(trX)} | held-out {len(teX)} | const-0 {base:.4f}")
    print("GBT raw 0.6126 | GBT hand+rel 0.5681\n")

    Xtr, Ptr = trX.astype(np.float32), trP.astype(np.float32)
    Vtr = trY.astype(np.float32)[:, None]
    Xte = teX.astype(np.float32)
    w = trW.astype(np.float32)
    m1, m2, m3 = ply <= 9, (ply >= 10) & (ply <= 21), ply >= 22

    print(f"{'arm':<20} {'ep':>3} {'train_v':>8} {'HELD-OUT':>9} {'vs c0':>7} "
          f"{'ply0-9':>8} {'ply10-21':>9} {'ply22+':>8}")
    print("-" * 78)
    for kind in ("muon", "adamw"):
        for ortho in (False, True):
            wd = 0.004
            tf.keras.utils.set_random_seed(0)
            model = build(kind, wd, ortho)
            tag = f"{kind} {'+ortho' if ortho else 'plain'}"
            best = (1e9, -1, None)
            for ep in range(1, EPOCHS + 1):
                h = model.fit(Xtr, (Ptr, Vtr), sample_weight=(w, w), batch_size=BATCH,
                              epochs=1, verbose=0, shuffle=True)
                pv = model.predict(Xte, batch_size=4096, verbose=0)[1][:, 0].astype(np.float64)
                tot = wmse(teY, pv, teW)
                if tot < best[0]:
                    best = (tot, ep, (wmse(teY[m1], pv[m1], teW[m1]),
                                      wmse(teY[m2], pv[m2], teW[m2]),
                                      wmse(teY[m3], pv[m3], teW[m3])))
                print(f"{tag if ep == 1 else '':<20} {ep:>3} "
                      f"{h.history['value_loss'][0]:>8.4f} {tot:>9.4f} {1-tot/base:>6.1%} "
                      f"{wmse(teY[m1],pv[m1],teW[m1]):>8.4f} "
                      f"{wmse(teY[m2],pv[m2],teW[m2]):>9.4f} "
                      f"{wmse(teY[m3],pv[m3],teW[m3]):>8.4f}", flush=True)
            b = best[2]
            print(f"{'  BEST':<20} {best[1]:>3} {'':>8} {best[0]:>9.4f} "
                  f"{1-best[0]/base:>6.1%} {b[0]:>8.4f} {b[1]:>9.4f} {b[2]:>8.4f}\n",
                  flush=True)


if __name__ == "__main__":
    main()
