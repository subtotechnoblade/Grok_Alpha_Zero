"""EXP 8 - LR up to 1e-3, AdamW vs MuonW, value head 16 filters / dense 32, 5 epochs.

Short by design: every arm so far peaked on value at epoch 2-4, so 5 epochs covers the
peak and the start of the decay, which is all the checkpoint criteria need.

Six LR arms at the stock wd 0.004, plus a wd 0.1 pair at the middle LR - MuonW's weight
decay has never been swept (every Muon arm ran the 0.004 default) and it is the only
regulariser Muon still has, since its update is scale-free and head width cannot slow it.

  GBT raw 0.6126 | GBT hand+rel 0.5681 | const-0 0.7691
  best so far: adamw d8 6e-4 -> 0.6401 (16.8% vs c0) @ep3, polKL 0.5233
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
EPOCHS = 5
FILTERS = 16
DENSE = 32
CSV = os.path.join(HERE, "exp8.csv")

#      tag                  opt      lr      wd
ARMS = [("adamw 3e-4",     "adamw", 3e-4, 0.004),
        ("adamw 6e-4",     "adamw", 6e-4, 0.004),
        ("adamw 1e-3",     "adamw", 1e-3, 0.004),
        ("muonw 3e-4",     "muon",  3e-4, 0.004),
        ("muonw 6e-4",     "muon",  6e-4, 0.004),
        ("muonw 1e-3",     "muon",  1e-3, 0.004),
        ("adamw 6e-4 wd.1", "adamw", 6e-4, 0.1),
        ("muonw 6e-4 wd.1", "muon",  6e-4, 0.1),
        # 2e-3 at wd 4e-3, the decay the best 1e-3 arm ran. The LR trend never turned
        # over at 1e-3, so this is the test for whether it does.
        ("adamw 2e-3",      "adamw", 2e-3, 0.004),
        ("muonw 2e-3",      "muon",  2e-3, 0.004)]


def compiled(kind, lr, wd):
    from Net.Alpha_Loss import Policy_Loss, Value_Loss
    from Net.Optimizers.adam import Adam
    from Net.Optimizers.muon import Muon
    from Connect4.config import optimizer_config

    m = make_model(value_filters=FILTERS, value_dense=DENSE)
    if kind == "muon":
        kw = dict(optimizer_config["kwargs"])
        kw.update(learning_rate=lr, weight_decay=wd)
        opt = Muon(**kw)
    else:
        opt = Adam(learning_rate=lr, weight_decay=wd, beta_1=0.9, beta_2=0.995)
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
    print(f"value: const-0 {v0:.4f} | GBT raw 0.6126\n")

    done = resume_counts(CSV, {a[0]: EPOCHS for a in ARMS})
    hdr = (f"{'arm':<18} {'ep':>3} {'VALUE':>8} {'vs c0':>7} {'ply22+':>8} | "
           f"{'polCE':>7} {'polKL':>7} {'top1':>7}")
    print(hdr)
    print("-" * len(hdr))

    for tag, kind, lr, wd in ARMS:
        if done.get(tag, 0) >= EPOCHS:
            print(f"{tag:<18} already complete, skipping", flush=True)
            continue
        tf.keras.utils.set_random_seed(0)
        model = compiled(kind, lr, wd)
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
            print(f"{tag if ep == 1 else '':<18} {ep:>3} {r['v']:>8.4f} "
                  f"{1-r['v']/v0:>6.1%} {r['vb'][2]:>8.4f} | {r['ce']:>7.4f} "
                  f"{r['kl']:>7.4f} {r['top1']:>6.1%}", flush=True)
        print(f"{'  BEST':<18} value {bv[0]:.4f} @ep{bv[1]}   "
              f"policyKL {bp[0]:.4f} @ep{bp[1]}\n", flush=True)
        del model
        tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()
