"""EXP 5 - LR sweep x optimizer x value-head width, and is the checkpoint criterion right.

Back to the stock setup (no dropout, no tricks) and change one thing at a time:

  value dense 32 -> 8   filters were shown not to matter (1/2/4/16 all within noise),
                        so the only real capacity knob left in the head is the dense
                        layer. Filters stay at 2, not 1, so the head keeps room for
                        later generations.
  adamw vs muon         both at every LR - their usable LR bands are not the same and
                        a conclusion drawn from one must not be assumed for the other.
  lr 3e-4 / 1e-4 / 3e-5 epochs scale roughly inversely so each LR gets a fair shot at
                        its own optimum instead of being scored while still underfit.

Every epoch of every arm is appended to exp5.csv as it happens, so a crash keeps the
data and the checkpoint-criterion analysis can be redone without retraining.

  GBT raw 0.6126 | GBT hand+rel 0.5681 | const-0 0.7691
"""
import os
import re
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.chdir(ROOT)

import numpy as np
import tensorflow as tf

from exp2_optimizer import load_split, dedup, wmse
from exp4_policy_value import make_model, scaled_adam, Scorer, VALUE_VARS

BATCH = 1024
CSV = os.path.join(HERE, "exp5.csv")

#      tag                        opt      lr      dense  vlr   epochs
ARMS = [("adamw d32 lr3e-4",     "adamw", 3e-4,    32,   1.0,   14),
        ("adamw d8  lr3e-4",     "adamw", 3e-4,     8,   1.0,   14),
        ("muon  d32 lr3e-4",     "muon",  3e-4,    32,   1.0,   14),
        ("muon  d8  lr3e-4",     "muon",  3e-4,     8,   1.0,   14),
        ("adamw d8  lr1e-4",     "adamw", 1e-4,     8,   1.0,   24),
        ("muon  d8  lr1e-4",     "muon",  1e-4,     8,   1.0,   24),
        ("adamw d8  lr3e-5",     "adamw", 3e-5,     8,   1.0,   30),
        ("muon  d8  lr3e-5",     "muon",  3e-5,     8,   1.0,   30),
        ("adamw d8  lr3e-4 vlr/5", "adamw", 3e-4,   8,   0.2,   14)]


def compiled(kind, lr, dense, vlr):
    from Net.Alpha_Loss import Policy_Loss, Value_Loss
    from Net.Optimizers.adam import Adam
    from Net.Optimizers.muon import Muon
    from Connect4.config import optimizer_config

    m = make_model(value_filters=2, value_dense=dense)
    if kind == "muon":
        kw = dict(optimizer_config["kwargs"])
        kw.update(learning_rate=lr, weight_decay=0.004)
        opt = Muon(**kw)
    elif vlr != 1.0:
        opt = scaled_adam([VALUE_VARS], vlr, learning_rate=lr, weight_decay=0.004,
                          beta_1=0.9, beta_2=0.995)
    else:
        opt = Adam(learning_rate=lr, weight_decay=0.004, beta_1=0.9, beta_2=0.995)
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
    steps = int(np.ceil(len(trX) / BATCH))

    print(f"train {len(trX)} unique | {steps} steps/ep | held-out {len(teX)}")
    print(f"value: const-0 {v0:.4f} | GBT raw 0.6126 | GBT hand+rel 0.5681")
    print(f"policy: entropy floor {score.H:.4f}\n")

    with open(CSV, "w") as f:
        f.write("arm,ep,value,ply0_9,ply10_21,ply22,polCE,polKL,top1,predstd\n")

    hdr = (f"{'arm':<24} {'ep':>3} {'VALUE':>8} {'vs c0':>7} {'ply22+':>8} | "
           f"{'polCE':>7} {'polKL':>7} {'top1':>7}")

    for tag, kind, lr, dense, vlr, epochs in ARMS:
        tf.keras.utils.set_random_seed(0)
        model = compiled(kind, lr, dense, vlr)
        print(hdr)
        print("-" * len(hdr))
        bv, bp = (1e9, -1), (1e9, -1)
        for ep in range(1, epochs + 1):
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
