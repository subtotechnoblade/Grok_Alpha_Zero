"""EXP 11 - lr 0.2, AdamW vs MuonW, 16 filters / dense 32, wd 4e-3, no orthograd.

0.2 is AlphaZero's initial SGD LR. Adam's step is lr*m/sqrt(v) ~ lr per parameter
regardless of gradient size, so 0.2 is an enormous step for it and divergence is a
real outcome, not a failure of the run - nan/inf rows are the result and are recorded
rather than crashed on.

  exp9 reference, identical otherwise:
    adamw 1e-2 -> 0.5490 (28.6% vs c0) @ep4, polKL 0.3562 @ep10
    muonw 1e-2 -> 0.6157 (19.9% vs c0) @ep2, polKL 0.3823 @ep9

  GBT raw 0.6126 | GBT hand+rel 0.5681 | const-0 0.7691
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
from exp9_high_lr import production_value_head

BATCH = 1024
EPOCHS = 10
LR = 0.2
WD = 4e-3
FILTERS = 16
DENSE = 32
CSV = os.path.join(HERE, "exp11.csv")

ARMS = [("adamw 0.2", "adamw"), ("muonw 0.2", "muon")]


def compiled(kind):
    from Net.Alpha_Loss import Policy_Loss, Value_Loss
    from Net.Optimizers.adam import Adam
    from Net.Optimizers.muon import Muon
    from Connect4.config import optimizer_config

    m = make_model(value_filters=FILTERS, value_dense=DENSE)
    if kind == "muon":
        kw = dict(optimizer_config["kwargs"])
        kw.update(learning_rate=LR, weight_decay=WD)
        opt = Muon(**kw)
    else:
        opt = Adam(learning_rate=LR, weight_decay=WD, beta_1=0.9, beta_2=0.995)
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

    try:
        pf, pd = production_value_head()
        note = "" if (pf, pd) == (FILTERS, DENSE) else "   <-- DIFFERENT, on purpose"
        prod = f"Connect4/Build_Model.py filters {pf} dense {pd}{note}"
    except Exception as e:
        prod = f"Connect4/Build_Model.py UNREADABLE ({type(e).__name__}: {e})"
    print(f"train {len(trX)} unique | held-out {len(teX)}")
    print(f"value head: THIS RUN filters {FILTERS} dense {DENSE}  |  {prod}")
    print(f"value: const-0 {v0:.4f} | GBT raw 0.6126 | lr {LR} | wd {WD} | no orthograd\n")

    done = resume_counts(CSV, {a[0]: EPOCHS for a in ARMS})
    hdr = (f"{'arm':<12} {'ep':>3} {'VALUE':>8} {'vs c0':>7} {'ply22+':>8} | "
           f"{'polCE':>7} {'polKL':>7} {'top1':>7}")
    print(hdr)
    print("-" * len(hdr))

    for tag, kind in ARMS:
        if done.get(tag, 0) >= EPOCHS:
            print(f"{tag:<12} already complete, skipping", flush=True)
            continue
        tf.keras.utils.set_random_seed(0)
        model = compiled(kind)
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
            print(f"{tag if ep == 1 else '':<12} {ep:>3} {r['v']:>8.4f} "
                  f"{1-r['v']/v0:>6.1%} {r['vb'][2]:>8.4f} | {r['ce']:>7.4f} "
                  f"{r['kl']:>7.4f} {r['top1']:>6.1%}", flush=True)
            if not np.isfinite(r["v"]):
                print(f"{'  DIVERGED':<12} non-finite at ep{ep}, stopping arm", flush=True)
                break
        print(f"{'  BEST':<12} value {bv[0]:.4f} @ep{bv[1]}   "
              f"policyKL {bp[0]:.4f} @ep{bp[1]}\n", flush=True)
        del model
        tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()
