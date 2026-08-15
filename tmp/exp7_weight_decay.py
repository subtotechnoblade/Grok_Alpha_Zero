"""EXP 7 - MuonW vs AdamW at matched weight decay, dense 32.

exp5 showed Muon's value head falls below const-0 and keeps going, and that shrinking
the head cannot regularise it (Muon's step is scale-free: 0.2*sqrt(max(d_out,d_in)) is
sqrt(84) for both Dense(84->32) and Dense(84->8), so both heads move at the same rate).
Weight decay is the regulariser Muon still has, and it was never swept for Muon - every
arm so far ran the 0.004 default.

Both optimizers decay decoupled as w -= w*lr*wd (muon.py apply_weight_decay, Keras
BaseOptimizer._apply_weight_decay), so wd means the same thing to both and the 2x2 is a
fair comparison.

Dense stays 32 - the 8-wide head is not being adopted, so it must not be confounded in.

  GBT raw 0.6126 | const-0 0.7691
"""
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.chdir(ROOT)

import numpy as np
import tensorflow as tf

from exp2_optimizer import load_split, dedup
from exp4_policy_value import make_model, Scorer
from exp6_lr_up import resume_counts

BATCH = 1024
LR = 3e-4
EPOCHS = 14
CSV = os.path.join(HERE, "exp7.csv")

#      tag                 opt      wd
ARMS = [("adamw wd0.004", "adamw", 0.004),
        ("adamw wd0.1",   "adamw", 0.1),
        ("muonw wd0.004", "muon",  0.004),
        ("muonw wd0.1",   "muon",  0.1)]


def compiled(kind, wd):
    from Net.Alpha_Loss import Policy_Loss, Value_Loss
    from Net.Optimizers.adam import Adam
    from Net.Optimizers.muon import Muon
    from Connect4.config import optimizer_config

    m = make_model(value_filters=2, value_dense=32)
    if kind == "muon":
        kw = dict(optimizer_config["kwargs"])
        kw.update(learning_rate=LR, weight_decay=wd)
        opt = Muon(**kw)
    else:
        opt = Adam(learning_rate=LR, weight_decay=wd, beta_1=0.9, beta_2=0.995)
    m.compile(optimizer=opt, loss={"policy": Policy_Loss(dtype="float32"),
                                   "value": Value_Loss(dtype="float32")})
    return m


def main():
    d = load_split()
    trX, trP, trY, trW = dedup(*d["tr"])
    teX, teP, teY, teW = dedup(*d["te"])
    ply = (teX[..., -1] != 0).reshape(len(teX), -1).sum(1)
    score = Scorer(teX.astype(np.float32), teP, teY, teW, ply)
    v0 = score.v0

    Xtr, Ptr = trX.astype(np.float32), trP.astype(np.float32)
    Vtr = trY.astype(np.float32)[:, None]
    w = trW.astype(np.float32)

    print(f"train {len(trX)} unique | held-out {len(teX)}")
    print(f"value: const-0 {v0:.4f} | GBT raw 0.6126 | lr {LR} | dense 32\n")

    done = resume_counts(CSV, {a[0]: EPOCHS for a in ARMS})

    hdr = (f"{'arm':<24} {'ep':>3} {'VALUE':>8} {'vs c0':>7} {'ply22+':>8} | "
           f"{'polCE':>7} {'polKL':>7} {'top1':>7}")

    for tag, kind, wd in ARMS:
        if done.get(tag, 0) >= EPOCHS:
            print(f"{tag:<24} already complete, skipping", flush=True)
            continue
        tf.keras.utils.set_random_seed(0)
        model = compiled(kind, wd)
        print(hdr)
        print("-" * len(hdr))
        bv, bp = (1e9, -1), (1e9, -1)
        for ep in range(1, EPOCHS + 1):
            model.fit(Xtr, (Ptr, Vtr), sample_weight=(w, w), batch_size=BATCH,
                      epochs=1, verbose=0, shuffle=True)
            r = score(model)
            bv, bp = min(bv, (r["v"], ep)), min(bp, (r["kl"], ep))
            with open(CSV, "a") as f:
                f.write(f"{tag},{ep},{r['v']:.6f},{r['vb'][0]:.6f},{r['vb'][1]:.6f},"
                        f"{r['vb'][2]:.6f},{r['ce']:.6f},{r['kl']:.6f},"
                        f"{r['top1']:.6f},{r['std']:.6f}\n")
            print(f"{tag if ep == 1 else '':<24} {ep:>3} {r['v']:>8.4f} "
                  f"{1-r['v']/v0:>6.1%} {r['vb'][2]:>8.4f} | {r['ce']:>7.4f} "
                  f"{r['kl']:>7.4f} {r['top1']:>6.1%}", flush=True)
        print(f"{'  BEST':<24} value {bv[0]:.4f} @ep{bv[1]}   "
              f"policyKL {bp[0]:.4f} @ep{bp[1]}\n", flush=True)
        del model
        tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()
