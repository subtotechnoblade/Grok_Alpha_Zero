"""Experiment H: does Orthograd help? Adam / AdamW / Muon, each with and without.

Short by design: 4 new arms x 2 learning rates x 5 epochs, 1 seed. The muon pair
already exists in results_cifar_sweep*.json (muon 0.7559, orthograd(muon) 0.6943,
both at 1e-1) so it is not re-run.

Two corrections baked in:
  - Orthograd wraps update_step, but keras applies decoupled weight decay in apply(),
    which never fires on a wrapped optimizer. So Orthograd(keras AdamW) silently has
    NO weight decay. WrapSafeAdam applies it in update_step instead, like Muon does.
  - "adam" vs "adamw" here is purely weight_decay 0 vs 1e-4, same class otherwise.

    python tmp/exp_h_ortho.py
"""
import os, sys, json, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tmp"))

import numpy as np
import tensorflow as tf
from keras.src import ops
for d in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(d, True)

from Net.Optimizers.ortho_grad import Orthograd
from exp_f_cifar import build_resnet, get_data, trunk_norm

LRS = [3e-3, 1e-2]
EPOCHS, SEED = 5, 7


class WrapSafeAdam(tf.keras.optimizers.Adam):
    """Adam with decoupled weight decay applied inside update_step, so it survives
    being wrapped by Orthograd. weight_decay=0.0 makes it plain Adam."""

    def __init__(self, learning_rate, wd=1e-4, **kw):
        super().__init__(learning_rate=learning_rate, beta_1=0.9, beta_2=0.995, **kw)
        self.wd = wd

    def update_step(self, gradient, variable, learning_rate):
        super().update_step(gradient, variable, learning_rate)
        if self.wd:
            lr = ops.cast(learning_rate, variable.dtype)
            variable.assign_sub(lr * ops.cast(self.wd, variable.dtype) * variable)


ARMS = [("adam",        lambda lr: WrapSafeAdam(lr, wd=0.0)),
        ("adam+ortho",   lambda lr: Orthograd(WrapSafeAdam(lr, wd=0.0))),
        ("adamw",        lambda lr: WrapSafeAdam(lr, wd=1e-4)),
        ("adamw+ortho",  lambda lr: Orthograd(WrapSafeAdam(lr, wd=1e-4)))]


def main():
    train, test = get_data()
    out, t0 = {}, time.time()
    print(f"{'arm':<14} {'lr':>7} {'test acc':>10} {'train acc':>11} {'|W| growth':>12} {'s':>6}")
    print("-" * 64)
    for label, make in ARMS:
        for lr in LRS:
            tf.keras.utils.set_random_seed(SEED)
            model = build_resnet()
            w0 = trunk_norm(model)
            model.compile(optimizer=make(lr), loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])
            s = time.time()
            h = model.fit(train, validation_data=test, epochs=EPOCHS, verbose=0)
            r = dict(best_acc=float(max(h.history["val_accuracy"])),
                     train_acc=float(h.history["accuracy"][-1]),
                     wgrowth=trunk_norm(model) / w0, secs=time.time() - s,
                     curve=[float(a) for a in h.history["val_accuracy"]])
            out[f"{label}@{lr:g}"] = r
            print(f"{label:<14} {lr:>7g} {r['best_acc']:>10.4f} {r['train_acc']:>11.4f} "
                  f"{r['wgrowth']:>11.2f}x {r['secs']:>6.0f}", flush=True)

    print("-" * 64)
    print(f"total {time.time() - t0:.0f}s\n")
    print("for reference, from the earlier sweeps (same model, 5 epochs, seed 7):")
    print(f"{'muon':<14} {0.1:>7g} {0.7559:>10.4f} {0.8138:>11.4f} {1.445:>11.2f}x")
    print(f"{'muon+ortho':<14} {0.1:>7g} {0.6943:>10.4f} {0.7919:>11.4f} {1.467:>11.2f}x")

    with open(os.path.join(ROOT, "tmp", "results_ortho.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
