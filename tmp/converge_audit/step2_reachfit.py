"""STEP 2 - what does it take to reach production's TRAIN fit?

Cells vary the three things that separate tmp/weight_sweep.py from production:

  init          scratch (the sweep) vs warm from the archive's own gen-N
                model.weights.h5 (production, pipeline.py:234)
  weight_decay  the sweep passed no weight_decay, so Muon's DEFAULT 0.004 applied.
                git HEAD's Connect4/config.py - the config both archived runs were
                produced under - sets 1e-4.  40x apart.
  lr / epochs   3e-4 x 10 was the sweep; production ran 8.  Both are extended to
                100 epochs here so the fit ceiling is visible, not guessed.

usage: step2_reachfit.py [run] [gen] [epochs]
"""
import os
import sys
import numpy as np
import tensorflow as tf

from common import (D, cbuild, build_and_compile, prep_eval, floors_for, arch_of, weights_path,
                    EpochProbe, wmse, dump, BATCH)

RUN = sys.argv[1] if len(sys.argv) > 1 else "run3"
GEN = int(sys.argv[2]) if len(sys.argv) > 2 else 12
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 100
TARGET = "y_blend"
TRAIN_DECAY, TEST_DECAY = 0.1, 0.01
# step0b_window.py pinned production's num_previous_generations at 5 for run3
# (Connect4/log.txt gen 5 collected gens 5,4,3,2,1).  The sweeps all used 4.
N_PREV = int(os.environ.get("N_PREV", "5"))

SWEEP_WD = None      # -> Muon default 0.004, what the sweeps actually got
PROD_WD = 1e-4       # git HEAD Connect4/config.py

#  name                             init      lr     wd         arch
# arch None = the run's own production architecture (run2 old, run3 current).
# "current" on run2 is the architecture the SWEEPS actually trained - a different
# net from the one that produced run2's data.
CELLS = [
    ("sweepARCH scratch lr3e-4 wd.004", "scratch", 3e-4, SWEEP_WD, "current"),
    ("sweep     scratch lr3e-4 wd.004", "scratch", 3e-4, SWEEP_WD, None),
    ("          scratch lr3e-4 wd1e-4", "scratch", 3e-4, PROD_WD,  None),
    ("PROD      warm    lr3e-4 wd1e-4", "warm",    3e-4, PROD_WD,  None),
    ("          warm    lr3e-4 wd.004", "warm",    3e-4, SWEEP_WD, None),
    ("          scratch lr1e-3 wd1e-4", "scratch", 1e-3, PROD_WD,  None),
    ("          warm    lr1e-3 wd1e-4", "warm",    1e-3, PROD_WD,  None),
]
if len(sys.argv) > 4:                     # optional cell filter, e.g. "PROD,sweep"
    pats = sys.argv[4].split(",")
    CELLS = [c for c in CELLS if any(p in c[0] for p in pats)]


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
    arch = arch_of(RUN)
    warm_from = weights_path(RUN, GEN)

    print(f"\n{RUN} gen {GEN} | arch {arch} | n_train {len(Xtr)} | {EPOCHS} epochs max")
    for sp, e in ev.items():
        print(f"  {sp:<6} n={len(e['y']):>6}  floor {np.average(e['floor'], weights=e['w']):.4f}  "
              f"const-0 {wmse(e['y'], np.zeros(len(e['y'])), e['w']):.4f}")
    print(f"  warm start = {warm_from}")

    out = {}
    for name, init, lr, wd, cell_arch in CELLS:
        a = cell_arch or arch
        if init == "warm" and a != arch:
            print(f"\n=== {name} === SKIPPED: no {a}-arch checkpoint exists for {RUN}")
            continue
        print(f"\n=== {name}  (arch {a}) ===", flush=True)
        tf.keras.utils.set_random_seed(0)
        model = build_and_compile(lr=lr, arch=a, weight_decay=wd,
                                  warm=(warm_from if init == "warm" else None))
        probe = EpochProbe(ev, dense_until=10, every=5, total=EPOCHS)
        model.fit(Xtr, (Ptr, Vtr), sample_weight=(wf, wf), batch_size=BATCH,
                  epochs=EPOCHS, verbose=0, shuffle=True, callbacks=[probe])
        out[name] = {"init": init, "lr": lr, "wd": wd, "arch": a, "hist": probe.hist}
        dump(f"step2_reachfit_{RUN}_g{GEN}_np{N_PREV}.json",
             {"run": RUN, "gen": GEN, "n_prev": N_PREV, "n_train": int(len(Xtr)), "cells": out})


if __name__ == "__main__":
    main()
