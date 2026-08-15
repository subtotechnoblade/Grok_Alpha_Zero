"""WarmupCosine step schedule vs the flat lr, 8 epochs, peak 1e-2, wd 4e-3, 16f/32d.

Same three optimizers, only the lr shape changes.

  flat-lr reference at 1e-2 (exp9/exp12):
    adamw       0.5490 (28.6%) @ep4, polKL 0.3562 @ep10, ep8 value 24.7%
    nadamw      0.5771 (25.0%) @ep4, polKL 0.3823 @ep10, ep8 value 16.9%
    muon+nadam  0.5885 (23.5%) @ep2, polKL 0.3730 @ep9,  ep8 value  9.2%
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
from Net.Schedules import WarmupCosine

BATCH = 1024
EPOCHS = 8
PEAK = 1e-2
WD = 4e-3
FILTERS = 16
DENSE = 32
CSV = os.path.join(HERE, "exp13.csv")

ARMS = [("adamw cos", "adam"), ("nadamw cos", "nadam"), ("muon+nadam cos", "muon")]


def compiled(kind, total_steps):
    from Net.Alpha_Loss import Policy_Loss, Value_Loss
    from Net.Optimizers.adam import Adam
    from Net.Optimizers.nadam import Nadam
    from Net.Optimizers.muon import Muon
    from Connect4.config import optimizer_config

    lr = WarmupCosine(PEAK, total_steps,
                      warmup_frac=optimizer_config["lr_warmup_frac"],
                      floor_frac=optimizer_config["lr_floor_frac"])
    m = make_model(value_filters=FILTERS, value_dense=DENSE)
    if kind == "muon":
        opt = Muon(learning_rate=lr, weight_decay=WD, use_nadam=True,
                   adam_beta_1=0.90, adam_beta_2=0.995, muon_beta=0.95,
                   exclude_layers=["eyes", "policy_1", "value_1"])
    elif kind == "nadam":
        opt = Nadam(learning_rate=lr, weight_decay=WD, beta_1=0.90, beta_2=0.995)
    else:
        opt = Adam(learning_rate=lr, weight_decay=WD, beta_1=0.90, beta_2=0.995)
    m.compile(optimizer=opt, loss={"policy": Policy_Loss(dtype="float32"),
                                   "value": Value_Loss(dtype="float32")})
    return m, lr


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
    total_steps = steps * EPOCHS

    print(f"train {len(trX)} unique | {steps} steps/epoch x {EPOCHS} = {total_steps} steps")
    print(f"value: const-0 {v0:.4f} | GBT raw 0.6126 | hand+rel 0.5681 | peak {PEAK} wd {WD}\n")

    done = resume_counts(CSV, {a[0]: EPOCHS for a in ARMS})
    hdr = (f"{'arm':<16} {'ep':>3} {'lr':>9} {'VALUE':>8} {'vs c0':>7} {'ply22+':>8} | "
           f"{'polKL':>7} {'top1':>7}")
    print(hdr)
    print("-" * len(hdr))

    for tag, kind in ARMS:
        if done.get(tag, 0) >= EPOCHS:
            print(f"{tag:<16} already complete, skipping", flush=True)
            continue
        tf.keras.utils.set_random_seed(0)
        model, sched = compiled(kind, total_steps)
        bv, bp = (1e9, -1), (1e9, -1)
        for ep in range(1, EPOCHS + 1):
            model.fit(Xtr, (Ptr, Vtr), sample_weight=(w, w), batch_size=BATCH,
                      epochs=1, verbose=0, shuffle=True)
            r = score(model)
            if np.isfinite(r["v"]):
                bv, bp = min(bv, (r["v"], ep)), min(bp, (r["kl"], ep))
            with open(CSV, "a") as f:
                f.write(f"{tag},{ep},{r['v']:.6f},{r['vb'][0]:.6f},{r['vb'][1]:.6f},"
                        f"{r['vb'][2]:.6f},{r['ce']:.6f},{r['kl']:.6f},"
                        f"{r['top1']:.6f},{r['std']:.6f}\n")
            print(f"{tag if ep == 1 else '':<16} {ep:>3} {float(sched(ep*steps)):>9.6f} "
                  f"{r['v']:>8.4f} {1-r['v']/v0:>6.1%} {r['vb'][2]:>8.4f} | "
                  f"{r['kl']:>7.4f} {r['top1']:>6.1%}", flush=True)
        print(f"{'  BEST':<16} value {bv[0]:.4f} @ep{bv[1]}   "
              f"policyKL {bp[0]:.4f} @ep{bp[1]}\n", flush=True)
        del model
        tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()
