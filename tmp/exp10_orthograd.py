"""EXP 10 - weight decay OFF, OrthoGrad ON, AdamW at 5e-3 and 1e-2.

No wd control arm: wd 3e-4 / 0.004 / 0.1 all measured identical in exp8, so switching it
off is a no-op and these two arms isolate OrthoGrad against exp9's matching arms.

  exp9 reference, same head, same epochs, wd 0.004, no ortho:
    adamw 5e-3 -> 0.5803 (24.6% vs c0) @ep4, polKL 0.3682 @ep9
    adamw 1e-2 -> 0.5490 (28.6% vs c0) @ep4, polKL 0.3562 @ep10

Excluded layers come from Connect4/config.py orthograd_exclude, not a constant here, and
the resulting mask is printed - orthogonalising an output head is what destroyed the
value head last time (weight norm 0.11 vs 6.01 after 20k steps), so which variables are
actually excluded has to be visible, not assumed.

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
WD = 0.0
FILTERS = 16
DENSE = 32      # matches exp9 so the only change vs its arms is wd-off + orthograd
CSV = os.path.join(HERE, "exp10.csv")

ARMS = [("adamw 5e-3 ortho", 5e-3),
        ("adamw 1e-2 ortho", 1e-2)]


def compiled(lr):
    from Net.Alpha_Loss import Policy_Loss, Value_Loss
    from Net.Optimizers.adam import Adam
    from Net.Optimizers.ortho_grad import Orthograd
    from Connect4.config import optimizer_config

    m = make_model(value_filters=FILTERS, value_dense=DENSE)
    exclude = optimizer_config.get("orthograd_exclude", ())
    opt = Orthograd(Adam(learning_rate=lr, weight_decay=WD, beta_1=0.9, beta_2=0.995),
                    exclude_layers=exclude)
    m.compile(optimizer=opt, loss={"policy": Policy_Loss(dtype="float32"),
                                   "value": Value_Loss(dtype="float32")})
    return m, exclude


def report_mask(model):
    """Which variables OrthoGrad actually touches. Built lazily, so run one step first."""
    opt = model.optimizer
    on = [v.path for v, m in zip(model.trainable_variables, opt.ortho_mask) if m]
    off = [v.path for v, m in zip(model.trainable_variables, opt.ortho_mask) if not m]
    print(f"  orthogonalised {len(on)}/{len(on)+len(off)} variables")
    print(f"  excluded 2D+: {[p for p in off if 'gamma' not in p and 'beta' not in p]}")


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

    try:                                  # a config readout must never kill a run
        pf, pd = production_value_head()
        note = "" if (pf, pd) == (FILTERS, DENSE) else "   <-- DIFFERENT, on purpose"
        prod = f"Connect4/Build_Model.py filters {pf} dense {pd}{note}"
    except Exception as e:
        prod = f"Connect4/Build_Model.py UNREADABLE ({type(e).__name__}: {e})"
    print(f"train {len(trX)} unique | held-out {len(teX)}")
    print(f"value head: THIS RUN filters {FILTERS} dense {DENSE}  |  {prod}")
    print(f"value: const-0 {v0:.4f} | GBT raw 0.6126 | wd {WD} | orthograd ON\n")

    done = resume_counts(CSV, {a[0]: EPOCHS for a in ARMS})
    hdr = (f"{'arm':<18} {'ep':>3} {'VALUE':>8} {'vs c0':>7} {'ply22+':>8} | "
           f"{'polCE':>7} {'polKL':>7} {'top1':>7}")

    for tag, lr in ARMS:
        if done.get(tag, 0) >= EPOCHS:
            print(f"{tag:<18} already complete, skipping", flush=True)
            continue
        tf.keras.utils.set_random_seed(0)
        model, exclude = compiled(lr)
        print(f"{tag}  exclude_layers={list(exclude)}")
        bv, bp = (1e9, -1), (1e9, -1)
        for ep in range(1, EPOCHS + 1):
            model.fit(Xtr, (Ptr, Vtr), sample_weight=(w, w), batch_size=BATCH,
                      epochs=1, verbose=0, shuffle=True)
            if ep == 1:
                report_mask(model)
                print(hdr)
                print("-" * len(hdr))
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
