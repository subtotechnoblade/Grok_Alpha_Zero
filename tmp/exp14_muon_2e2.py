"""Muon+Nadam at 2e-2, flat lr, wd 4e-3, 16 filters / 32 dense, no orthograd.

  muon+nadam flat reference:
    1e-2 -> 0.5885 (23.5%) @ep2, polKL 0.3730 @ep9, ep8 value 9.2%
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
EPOCHS = 8
LR = 2e-2
WD = 4e-3
FILTERS = 16
DENSE = 32
CSV = os.path.join(HERE, "exp14.csv")

ARMS = [("muon+nadam 2e-2", "muon")]


def compiled(kind):
    from Net.Alpha_Loss import Policy_Loss, Value_Loss
    from Net.Optimizers.muon import Muon

    m = make_model(value_filters=FILTERS, value_dense=DENSE)
    opt = Muon(learning_rate=LR, weight_decay=WD, use_nadam=True,
               adam_beta_1=0.90, adam_beta_2=0.995, muon_beta=0.95,
               exclude_layers=["eyes", "policy_1", "value_1"])
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

    print(f"train {len(trX)} unique | held-out {len(teX)} | filters {FILTERS} dense {DENSE}")
    print(f"value: const-0 {v0:.4f} | GBT raw 0.6126 | hand+rel 0.5681 | lr {LR} wd {WD}\n")

    done = resume_counts(CSV, {a[0]: EPOCHS for a in ARMS})
    hdr = (f"{'arm':<18} {'ep':>3} {'VALUE':>8} {'vs c0':>7} {'ply22+':>8} | "
           f"{'polKL':>7} {'top1':>7}")
    print(hdr)
    print("-" * len(hdr))

    for tag, kind in ARMS:
        if done.get(tag, 0) >= EPOCHS:
            print(f"{tag:<18} already complete, skipping", flush=True)
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
            print(f"{tag if ep == 1 else '':<18} {ep:>3} {r['v']:>8.4f} "
                  f"{1-r['v']/v0:>6.1%} {r['vb'][2]:>8.4f} | {r['kl']:>7.4f} "
                  f"{r['top1']:>6.1%}", flush=True)
            if not np.isfinite(r["v"]):
                print(f"{'  DIVERGED':<18} non-finite at ep{ep}, stopping arm", flush=True)
                break
        print(f"{'  BEST':<18} value {bv[0]:.4f} @ep{bv[1]}   "
              f"policyKL {bp[0]:.4f} @ep{bp[1]}\n", flush=True)
        del model
        tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()
