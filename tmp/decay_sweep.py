"""Does concentrating the training window on RECENT generations reduce deploy-time error?

Measured problem: model N is trained on gens N-4..N-1 but PLAYS gen N. On its own
generation it scores ~20x the noise floor; on the generations it trained on, ~1.2x.
That is an off-policy staleness gap. train_decay controls how much of each older
generation is kept, so it is the direct knob on staleness.

Production today is nearly FLAT (subtractive 0.1 -> 100/90/80/70/60/50% of each
generation's games). This sweeps the shape and scores every arm on `nxt` — gen N
itself, i.e. the deployment distribution.

Two confounds this controls for:
  * steeper decay also means LESS DATA, so the arms are not equal-sized. n_train is
    reported for every arm; a win that tracks n_train is a data effect, not a
    freshness effect.
  * the `newest_only` arm is the extreme: train on gen N-1 alone.

  usage: decay_sweep.py [run] [gen] [target] [epochs]
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
from weight_sweep import build_and_compile

RUN = sys.argv[1] if len(sys.argv) > 1 else "run2"
GEN = int(sys.argv[2]) if len(sys.argv) > 2 else 12
TARGET = sys.argv[3] if len(sys.argv) > 3 else "y_blend"
EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 10
BATCH = 1024
SEEDS = (0, 1)          # the weighting sweep showed ~0.06 of seed noise; average it out

# name -> kwargs for data.build. n_prev=4 keeps the window LENGTH fixed at 4 gens so
# only the SHAPE varies; 'newest_only' shortens it deliberately as the extreme point.
ARMS = {
    "flat_0.1 (production)": dict(n_prev=4, train_decay=0.1),
    "medium_0.2":            dict(n_prev=4, train_decay=0.2),
    "steep_0.3":             dict(n_prev=4, train_decay=0.3),
    "cliff_0.5":             dict(n_prev=4, train_decay=0.5),
    "newest_only":           dict(n_prev=1, train_decay=1.0),
}


def main():
    floors = json.load(open(os.path.join(HERE, "learnability",
                                         f"floorfull_{RUN}_g{GEN}.json")))
    tgt = "blend" if TARGET == "y_blend" else "z"

    # Evaluation splits come from ONE canonical build so every arm is scored on
    # identical data; only the TRAIN side changes between arms.
    base = D.get(RUN, GEN)
    ev = {}
    for split in ("nxt", "val"):
        s = base[split]
        ply = ply_of(s["X"])
        ev[split] = dict(X=s["X"].astype(np.float32), y=s[TARGET], w=s["w"], ply=ply,
                         floor=position_floor(ply, s["count"], floors, tgt))

    print(f"{RUN} gen {GEN} | target {TARGET} | {EPOCHS} epochs | seeds {SEEDS}")
    for split in ("nxt", "val"):
        e = ev[split]
        print(f"  {split}: floor {np.average(e['floor'], weights=e['w']):.4f}   "
              f"const-0 {wmse(e['y'], np.zeros(len(e['y'])), e['w']):.4f}   "
              f"({len(e['X'])} positions)")

    hdr = (f"\n{'arm':<22} {'n_train':>8} | {'NXT':>8} {'nxt/fl':>7} {'nxt end':>8} "
           f"{'end/fl':>7} | {'VAL':>8} {'val/fl':>7}")
    print(hdr)
    print("-" * (len(hdr) - 1))

    results = {}
    for name, kw in ARMS.items():
        d = D.build(RUN, GEN, verbose=False, next_gen_holdout=False, **kw)
        tr = d["train"]
        Xtr = tr["X"].astype(np.float32)
        Ptr = tr["policy"].astype(np.float32)
        Vtr = tr[TARGET].astype(np.float32)[:, None]
        wf = tr["w"].astype(np.float32)

        acc = {k: [] for k in ("nxt", "nxt_end", "val")}
        for seed in SEEDS:
            tf.keras.utils.set_random_seed(seed)
            model = build_and_compile()
            model.fit(Xtr, (Ptr, Vtr), sample_weight=(wf, wf), batch_size=BATCH,
                      epochs=EPOCHS, verbose=0, shuffle=True)
            for split in ("nxt", "val"):
                e = ev[split]
                pv = model.predict(e["X"], batch_size=4096, verbose=0)[1][:, 0]
                pv = pv.astype(np.float64)
                acc[split].append(wmse(e["y"], pv, e["w"]))
                if split == "nxt":
                    m = e["ply"] >= 22
                    acc["nxt_end"].append(wmse(e["y"][m], pv[m], e["w"][m]))

        nfl = float(np.average(ev["nxt"]["floor"], weights=ev["nxt"]["w"]))
        m = ev["nxt"]["ply"] >= 22
        efl = float(np.average(ev["nxt"]["floor"][m], weights=ev["nxt"]["w"][m]))
        vfl = float(np.average(ev["val"]["floor"], weights=ev["val"]["w"]))
        r = {k: float(np.mean(v)) for k, v in acc.items()}
        r["n_train"] = int(len(Xtr))
        r["nxt_sd"] = float(np.std(acc["nxt"]))
        results[name] = r
        print(f"{name:<22} {r['n_train']:>8} | {r['nxt']:>8.4f} {r['nxt']/nfl:>6.1f}x "
              f"{r['nxt_end']:>8.4f} {r['nxt_end']/efl:>6.1f}x | "
              f"{r['val']:>8.4f} {r['val']/vfl:>6.1f}x", flush=True)

    sds = [v["nxt_sd"] for v in results.values()]
    print(f"\nseed-to-seed sd on NXT, per arm: "
          f"{', '.join(f'{s:.4f}' for s in sds)}  (treat smaller gaps as noise)")
    with open(os.path.join(HERE, f"decay_sweep_{RUN}_g{GEN}_{tgt}.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
