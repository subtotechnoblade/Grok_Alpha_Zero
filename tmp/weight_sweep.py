"""ITEM 3 - does the dedup sample-weighting scheme matter for the VALUE head?

dedup_average does two separable things:
  (a) merge byte-identical states and AVERAGE their targets  -> noise reduction, good
  (b) use the occurrence count as the sample weight          -> reproduces the raw
      data distribution. This is the part under test.

Occurrence count is anti-correlated with learnability: openings repeat thousands of
times and are nearly unlearnable; endgames occur once and are nearly deterministic.
So (b) may be pouring gradient into the positions where nothing can be learned.

Schemes (all renormalised to mean 1, so the effective LR scale is unchanged):
  count   w = n            production today, == training on the raw duplicated set
  uniform w = 1            every distinct position equally important
  sqrt    w = sqrt(n)      gentle compression, keeps frequency ordering
  cap20   w = min(n, 20)   kills the 1500x domination, keeps ordering elsewhere
  invvar  w = n / (sigma^2(ply)/n + eps)   inverse variance of the DEDUPED target,
                                           i.e. trust positions whose target you know

Judged on `nxt` (the NEXT generation, produced by a newer net) rather than on the
in-window val split, because the in-window split massively understates the error on
the distribution the search will actually meet.
"""
import json
import os
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "learnability"))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import tensorflow as tf

import data as D
from valmetrics import ply_of, wmse, position_floor

RUN = sys.argv[1] if len(sys.argv) > 1 else "run2"
GEN = int(sys.argv[2]) if len(sys.argv) > 2 else 12
TARGET = sys.argv[3] if len(sys.argv) > 3 else "y_blend"
EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 10
BATCH = 1024


def norm1(w):
    return w * (len(w) / w.sum())


def schemes(count, ply, floors, tgt):
    n = count.astype(np.float64)
    sig = position_floor(ply, np.ones_like(count), floors, tgt)  # per-sample sigma^2
    return {
        "count":   norm1(n),
        "uniform": norm1(np.ones_like(n)),
        "sqrt":    norm1(np.sqrt(n)),
        "cap20":   norm1(np.minimum(n, 20.0)),
        "invvar":  norm1(n / (sig / n + 1e-3)),
    }


def build_and_compile():
    from Connect4.Build_Model import build_model
    from Connect4.game import Connect4
    from Connect4.config import build_config, train_config, optimizer_config
    from Net.Alpha_Loss import Policy_Loss, Value_Loss

    g = Connect4()
    model = build_model(g.get_input_state().shape, g.policy_shape,
                        (build_config, train_config, optimizer_config))
    from Net.Optimizers.adam import Adam
    from Net.Optimizers.nadam import Nadam
    from Net.Optimizers.muon import Muon
    opt_cls = {"adam": Adam, "nadam": Nadam,
               "muon": Muon}[optimizer_config["optimizer"].lower()]
    kwargs = dict(optimizer_config["kwargs"])
    kwargs["learning_rate"] = 3e-4          # lr_schedule(gen 0); a callable can't be passed
    opt = opt_cls(**kwargs)                 # NO orthograd, NO grokfast - decided config
    model.compile(optimizer=opt,
                  loss={"policy": Policy_Loss(dtype="float32"),
                        "value": Value_Loss(dtype="float32")})
    return model


def main():
    d = D.get(RUN, GEN)
    floors = json.load(open(os.path.join(HERE, "learnability",
                                         f"floorfull_{RUN}_g{GEN}.json")))
    tgt = "blend" if TARGET == "y_blend" else "z"
    tr, nx, va = d["train"], d["nxt"], d["val"]

    ply_tr = ply_of(tr["X"])
    ws = schemes(tr["count"], ply_tr, floors, tgt)

    Xtr = tr["X"].astype(np.float32)
    Ptr = tr["policy"].astype(np.float32)
    Vtr = tr[TARGET].astype(np.float32)[:, None]

    for name in ("nxt", "val"):
        s = d[name]
        s["_ply"] = ply_of(s["X"])
        s["_floor"] = position_floor(s["_ply"], s["count"], floors, tgt)
        s["_Xf"] = s["X"].astype(np.float32)

    print(f"{RUN} gen {GEN} | target {TARGET} | {EPOCHS} epochs | "
          f"train {len(Xtr)} unique")
    print(f"floors: nxt {np.average(nx['_floor'], weights=nx['w']):.4f}   "
          f"val {np.average(va['_floor'], weights=va['w']):.4f}")
    print(f"const-0: nxt {wmse(nx[TARGET], np.zeros(len(nx['X'])), nx['w']):.4f}   "
          f"val {wmse(va[TARGET], np.zeros(len(va['X'])), va['w']):.4f}\n")

    hdr = (f"{'scheme':<9} {'max w':>7} | {'NXT all':>8} {'nxt/fl':>7} "
           f"{'nxt end':>8} {'end/fl':>7} {'nxt unseen':>11} | {'VAL all':>8} {'val/fl':>7}")
    print(hdr)
    print("-" * len(hdr))

    results = {}
    for name, w in ws.items():
        tf.keras.utils.set_random_seed(0)
        model = build_and_compile()
        t0 = time.time()
        # tuple form, exactly like Dataloader.__getitem__ returns
        # (states, (policies, values), (weights, weights))
        wf = w.astype(np.float32)
        model.fit(Xtr, (Ptr, Vtr), sample_weight=(wf, wf),
                  batch_size=BATCH, epochs=EPOCHS, verbose=0, shuffle=True)
        row = {"train_s": time.time() - t0, "max_w": float(w.max())}
        for split in ("nxt", "val"):
            s = d[split]
            pv = model.predict(s["_Xf"], batch_size=2048, verbose=0)[1][:, 0]
            pv = pv.astype(np.float64)
            y, ww = s[TARGET], s["w"]
            row[f"{split}_all"] = wmse(y, pv, ww)
            row[f"{split}_fl"] = float(np.average(s["_floor"], weights=ww))
            m = s["_ply"] >= 22
            row[f"{split}_end"] = wmse(y[m], pv[m], ww[m])
            row[f"{split}_endfl"] = float(np.average(s["_floor"][m], weights=ww[m]))
            if split == "nxt":
                trk = {x.tobytes() for x in tr["X"]}
                un = np.array([x.tobytes() not in trk for x in s["X"]])
                row["nxt_unseen"] = wmse(y[un], pv[un], ww[un])
        results[name] = row
        print(f"{name:<9} {row['max_w']:>7.1f} | {row['nxt_all']:>8.4f} "
              f"{row['nxt_all']/row['nxt_fl']:>6.1f}x {row['nxt_end']:>8.4f} "
              f"{row['nxt_end']/row['nxt_endfl']:>6.1f}x {row['nxt_unseen']:>11.4f} | "
              f"{row['val_all']:>8.4f} {row['val_all']/row['val_fl']:>6.1f}x", flush=True)

    with open(os.path.join(HERE, f"weight_sweep_{RUN}_g{GEN}_{tgt}.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
