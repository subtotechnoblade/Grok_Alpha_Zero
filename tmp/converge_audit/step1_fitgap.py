"""STEP 1 - how far from production fit were the sweep models?

Trains EXACTLY the way tmp/weight_sweep.py did (from scratch, 10 epochs, lr 3e-4,
Muon, no orthograd, count weights, batch 1024) and records, per epoch:

    keras_train_value_loss   the number production's log.txt prints
    wmse_train               full-pass weighted value MSE on the train split
    wmse_val                 in-window holdout
    wmse_nxt                 next generation

The sweeps reported only the last two.  wmse_train is the missing number.

usage: step1_fitgap.py [run] [gen] [epochs]
"""
import sys
import numpy as np
import tensorflow as tf

from common import (D, cbuild, build_and_compile, prep_eval, floors_for, arch_of,
                    EpochProbe, wmse, dump, BATCH)

RUN = sys.argv[1] if len(sys.argv) > 1 else "run2"
GEN = int(sys.argv[2]) if len(sys.argv) > 2 else 12
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
TARGET = "y_blend"
SEEDS = (0, 1)
# 0.2/0.02 == data.get()'s defaults == what weight_sweep.py actually trained on.
# 0.1/0.01 == git HEAD Connect4/config.py == what run2/run3 production used.
TRAIN_DECAY = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
TEST_DECAY = float(sys.argv[5]) if len(sys.argv) > 5 else 0.02
N_PREV = int(sys.argv[6]) if len(sys.argv) > 6 else 4


def main():
    d = cbuild(RUN, GEN, n_prev=N_PREV, train_decay=TRAIN_DECAY, test_decay=TEST_DECAY,
               next_gen_holdout=True, verbose=True, seed=0)
    floors = floors_for(RUN, GEN)
    ev = prep_eval(d, ("train", "val", "nxt"), TARGET, floors, "blend")

    tr = d["train"]
    Xtr = tr["X"].astype(np.float32)
    Ptr = tr["policy"].astype(np.float32)
    Vtr = tr[TARGET].astype(np.float32)[:, None]
    wf = tr["w"].astype(np.float32)

    print(f"\n{RUN} gen {GEN} | arch {arch_of(RUN)} | scratch | {EPOCHS} epochs | lr 3e-4 | "
          f"n_train {len(Xtr)}")
    for sp, e in ev.items():
        print(f"  {sp:<6} n={len(e['y']):>6}  floor {np.average(e['floor'], weights=e['w']):.4f}  "
              f"const-0 {wmse(e['y'], np.zeros(len(e['y'])), e['w']):.4f}")

    runs = {}
    for seed in SEEDS:
        print(f"\n  --- seed {seed} ---")
        tf.keras.utils.set_random_seed(seed)
        model = build_and_compile(lr=3e-4, arch=arch_of(RUN))
        probe = EpochProbe(ev)
        model.fit(Xtr, (Ptr, Vtr), sample_weight=(wf, wf), batch_size=BATCH,
                  epochs=EPOCHS, verbose=0, shuffle=True, callbacks=[probe])
        runs[seed] = probe.hist

    dump(f"step1_fitgap_{RUN}_g{GEN}_e{EPOCHS}_td{TRAIN_DECAY}_np{N_PREV}.json",
         {"run": RUN, "gen": GEN, "epochs": EPOCHS, "n_train": int(len(Xtr)),
          "train_decay": TRAIN_DECAY, "n_prev": N_PREV,
          "seeds": {str(k): v for k, v in runs.items()}})


if __name__ == "__main__":
    main()
