"""Experiment I: project the UPDATE instead of the gradient. Does the mechanism come back?

4 runs only. The matching baselines (adam, muon) and the pre-projection arms are already
measured at the same seed / epochs / model, so they are printed for reference not re-run.

The number that decides it is the ||W|| growth column, not accuracy: stock Orthograd moved
it 1.84x -> 1.76x for adam and 1.45x -> 1.47x for muon, i.e. did nothing.

    python tmp/exp_i_post_ortho.py
"""
import os, sys, json, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tmp"))

import numpy as np
import tensorflow as tf
for d in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(d, True)

from Net.Optimizers.muon import Muon
from exp_f_cifar import build_resnet, get_data, trunk_norm, MUON_KWARGS
from exp_h_ortho import WrapSafeAdam
from post_ortho import PostOrthograd

EPOCHS, SEED = 5, 7
ARMS = [("adam+post", 3e-3, lambda lr: PostOrthograd(WrapSafeAdam(lr, wd=0.0))),
        ("adam+post", 1e-2, lambda lr: PostOrthograd(WrapSafeAdam(lr, wd=0.0))),
        ("muon+post", 3e-2, lambda lr: PostOrthograd(Muon(learning_rate=lr, **MUON_KWARGS))),
        ("muon+post", 1e-1, lambda lr: PostOrthograd(Muon(learning_rate=lr, **MUON_KWARGS)))]

REFERENCE = [("adam", 3e-3, 0.7023, 0.7949, 1.84), ("adam", 1e-2, 0.6519, 0.7456, 4.59),
             ("adam+ortho(grad)", 3e-3, 0.7082, 0.7817, 1.76),
             ("adam+ortho(grad)", 1e-2, 0.6425, 0.7422, 4.20),
             ("muon", 3e-2, 0.6545, 0.7693, 1.10), ("muon", 1e-1, 0.7559, 0.8138, 1.45),
             ("muon+ortho(grad)", 3e-2, 0.6694, 0.7389, 1.06),
             ("muon+ortho(grad)", 1e-1, 0.6943, 0.7919, 1.47)]


def main():
    train, test = get_data()
    out, t0 = {}, time.time()
    print(f"{'arm':<20} {'lr':>7} {'test acc':>10} {'train acc':>11} {'|W| growth':>12} {'s':>6}")
    print("-" * 70)
    for label, lr, make in ARMS:
        tf.keras.utils.set_random_seed(SEED)
        model = build_resnet()
        w0 = trunk_norm(model)
        model.compile(optimizer=make(lr), loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        s = time.time()
        h = model.fit(train, validation_data=test, epochs=EPOCHS, verbose=0)
        r = dict(best_acc=float(max(h.history["val_accuracy"])),
                 train_acc=float(h.history["accuracy"][-1]),
                 wgrowth=trunk_norm(model) / w0, secs=time.time() - s)
        out[f"{label}@{lr:g}"] = r
        print(f"{label:<20} {lr:>7g} {r['best_acc']:>10.4f} {r['train_acc']:>11.4f} "
              f"{r['wgrowth']:>11.2f}x {r['secs']:>6.0f}", flush=True)

    print("-" * 70)
    print(f"total {time.time() - t0:.0f}s\n\nalready measured, same seed / epochs / model:")
    for label, lr, acc, tacc, wg in REFERENCE:
        print(f"{label:<20} {lr:>7g} {acc:>10.4f} {tacc:>11.4f} {wg:>11.2f}x")

    with open(os.path.join(ROOT, "tmp", "results_post_ortho.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
