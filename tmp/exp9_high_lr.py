"""EXP 9 - 5e-3 and 1e-2, AdamW vs MuonW, value head 16 filters / dense 8, 10 epochs.

1e-2 is AlphaGo Zero's initial LR, but theirs was SGD+momentum: the step there is lr*g
and scales with the gradient, where Adam's is lr*m/sqrt(v) ~ lr per parameter. The
numbers are not in the same units, so this is an empirical test, not a port of their
setting.

2e-3 is repeated here as the bridge arm. exp8 ran 2e-3 at dense 32 and the head is now
dense 8, so without it a change could be the width rather than the LR. It also answers
whether AdamW 2e-3's flat value curve survives past 5 epochs, which 5 epochs could not
show.

  GBT raw 0.6126 | GBT hand+rel 0.5681 | const-0 0.7691
  best so far: adamw 2e-3 dense32 -> 0.6149 (20.1% vs c0) @ep2, polKL 0.4192 @ep5
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
EPOCHS = 10
WD = 0.004
FILTERS = 16
DENSE = 32      # deliberately NOT Build_Model's 8: exp8 ran 32, and matching it is what
                # keeps this a clean LR comparison rather than a width-vs-LR confound
CSV = os.path.join(HERE, "exp9.csv")

#      tag             opt      lr
ARMS = [("adamw 5e-3", "adamw", 5e-3),
        ("adamw 1e-2", "adamw", 1e-2),
        ("muonw 5e-3", "muon",  5e-3),
        ("muonw 1e-2", "muon",  1e-2)]


def production_value_head():
    """Value-head filters and dense width read off the real Connect4/Build_Model.

    These experiments rebuild the net rather than importing build_model (the head has to
    be parameterised), so a hand-set constant here can silently stop matching the config
    actually being run - it already did once, exp8 ran dense 32 against a dense 8 config.
    Read it instead of assuming it.
    """
    from Connect4.Build_Model import build_model
    from Connect4.game import Connect4
    from Connect4.config import build_config, train_config, optimizer_config

    g = Connect4()
    m = build_model(g.get_input_state().shape, g.policy_shape,
                    (build_config, train_config, optimizer_config))

    # Walk the graph back from value_1 rather than slicing m.layers by position:
    # list order is topological but interleaves the two heads, so a positional slice
    # picks up policy_1 and reports the policy width as the value width.
    produced_by = {}
    for layer in m.layers:
        for t in (layer.output if isinstance(layer.output, list) else [layer.output]):
            produced_by[id(t)] = layer

    chain, cur = [], m.get_layer("value_1")
    while cur is not None:
        ins = cur.input if isinstance(cur.input, list) else [cur.input]
        if not ins:                      # InputLayer: .input is [], end of the walk
            break
        cur = produced_by.get(id(ins[0]))
        if cur is not None:
            chain.append(cur)

    filters = next((l.filters for l in chain if isinstance(l, tf.keras.layers.Conv2D)), None)
    dense = next((l.units for l in chain if isinstance(l, tf.keras.layers.Dense)), None)
    return filters, dense


def compiled(kind, lr, filters, dense):
    from Net.Alpha_Loss import Policy_Loss, Value_Loss
    from Net.Optimizers.adam import Adam
    from Net.Optimizers.muon import Muon
    from Connect4.config import optimizer_config

    m = make_model(value_filters=filters, value_dense=dense)
    if kind == "muon":
        kw = dict(optimizer_config["kwargs"])
        kw.update(learning_rate=lr, weight_decay=WD)
        opt = Muon(**kw)
    else:
        opt = Adam(learning_rate=lr, weight_decay=WD, beta_1=0.9, beta_2=0.995)
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

    pf, pd = production_value_head()
    print(f"train {len(trX)} unique | held-out {len(teX)}")
    print(f"value head: THIS RUN filters {FILTERS} dense {DENSE}  |  "
          f"Connect4/Build_Model.py filters {pf} dense {pd}"
          + ("" if (pf, pd) == (FILTERS, DENSE) else "   <-- DIFFERENT, on purpose"))
    print(f"value: const-0 {v0:.4f} | GBT raw 0.6126 | wd {WD}\n")

    done = resume_counts(CSV, {a[0]: EPOCHS for a in ARMS})
    hdr = (f"{'arm':<14} {'ep':>3} {'VALUE':>8} {'vs c0':>7} {'ply22+':>8} | "
           f"{'polCE':>7} {'polKL':>7} {'top1':>7}")
    print(hdr)
    print("-" * len(hdr))

    for tag, kind, lr in ARMS:
        if done.get(tag, 0) >= EPOCHS:
            print(f"{tag:<14} already complete, skipping", flush=True)
            continue
        tf.keras.utils.set_random_seed(0)
        model = compiled(kind, lr, FILTERS, DENSE)
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
            print(f"{tag if ep == 1 else '':<14} {ep:>3} {r['v']:>8.4f} "
                  f"{1-r['v']/v0:>6.1%} {r['vb'][2]:>8.4f} | {r['ce']:>7.4f} "
                  f"{r['kl']:>7.4f} {r['top1']:>6.1%}", flush=True)
        print(f"{'  BEST':<14} value {bv[0]:.4f} @ep{bv[1]}   "
              f"policyKL {bp[0]:.4f} @ep{bp[1]}\n", flush=True)
        del model
        tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()
