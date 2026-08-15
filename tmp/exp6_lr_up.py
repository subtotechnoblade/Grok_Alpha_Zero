"""EXP 6 - LR sweep UPWARD, the direction exp5 actually pointed.

exp5 swept 3e-4 -> 1e-4 -> 3e-5 and every step down was worse on BOTH heads for BOTH
optimizers, matched at equal policy quality. Lower LR did not prevent the value
collapse, it only stretched it over more epochs and landed at a worse optimum. If the
mechanism is the usual one - big steps cannot settle into the narrow memorising minima
that small steps fall into - then the optimum is above 3e-4 and was never sampled,
because 3e-4 was the top of the range that was ever tried.

So: 6e-4 / 1e-3 / 3e-3, both optimizers, and one d32 arm at 1e-3 to check whether the
dense-8 advantage survives at a higher LR or is specific to 3e-4.

Compared at EQUAL POLICY QUALITY, never at equal epoch - different step sizes travel
different distances per epoch, so the epoch axis is not comparable across LRs.

  GBT raw 0.6126 | const-0 0.7691
  best so far: adamw d8 3e-4 -> value 13.3% vs c0 @ep4, polKL 0.4963 @ep14
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
from exp4_policy_value import Scorer
from exp5_lr_dense_opt import compiled

BATCH = 1024
CSV = os.path.join(HERE, "exp6.csv")
HEAD = "arm,ep,value,ply0_9,ply10_21,ply22,polCE,polKL,top1,predstd\n"


def resume_counts(csv, expected):
    """Arms already complete. The WSL dxgkrnl driver faults mid-run (dxgvmbus.c:3095)
    and kills the process, so finished arms must survive a restart.

    Rows of a PARTIAL arm are dropped, not kept: that arm reruns from epoch 1 and would
    otherwise append a second set of rows on top of the truncated first set.
    """
    if not os.path.exists(csv):
        with open(csv, "w") as f:
            f.write(HEAD)
        return {}
    rows = {}
    with open(csv) as f:
        next(f, None)
        for line in f:
            p = line.split(",")
            if len(p) > 1:
                rows.setdefault(p[0], []).append(line)
    done = {t: len(v) for t, v in rows.items() if len(v) >= expected.get(t, 1 << 30)}
    with open(csv, "w") as f:
        f.write(HEAD)
        for t in done:
            f.writelines(rows[t])
    return done

#      tag                  opt      lr     dense  vlr  epochs
ARMS = [("adamw d8  lr6e-4", "adamw", 6e-4,   8,  1.0,  12),
        ("adamw d8  lr1e-3", "adamw", 1e-3,   8,  1.0,  12),
        ("adamw d8  lr3e-3", "adamw", 3e-3,   8,  1.0,  12),
        ("adamw d32 lr1e-3", "adamw", 1e-3,  32,  1.0,  12),
        ("muon  d8  lr1e-3", "muon",  1e-3,   8,  1.0,  12),
        ("muon  d8  lr3e-3", "muon",  3e-3,   8,  1.0,  12)]


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
    print(f"value: const-0 {v0:.4f} | GBT raw 0.6126\n")

    done = resume_counts(CSV, {a[0]: a[5] for a in ARMS})

    hdr = (f"{'arm':<24} {'ep':>3} {'VALUE':>8} {'vs c0':>7} {'ply22+':>8} | "
           f"{'polCE':>7} {'polKL':>7} {'top1':>7}")

    for tag, kind, lr, dense, vlr, epochs in ARMS:
        if done.get(tag, 0) >= epochs:
            print(f"{tag:<24} already complete, skipping", flush=True)
            continue
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
