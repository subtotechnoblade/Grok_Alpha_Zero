"""Where does the time go: the data pipeline, or Newton-Schulz in bfloat16?

The RTX 2060 is Turing (compute 7.5). bfloat16 first got hardware support on Ampere,
so bf16 matmuls here are emulated. Muon casts to bf16 inside zeropower_via_newtonschulz5.

    python tmp/timing.py
"""
import os, sys, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import tensorflow as tf
from keras.src import ops
for d in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(d, True)

from Net.Optimizers.muon import Muon
sys.path.insert(0, os.path.join(ROOT, "tmp"))
from exp_f_cifar import build_resnet, get_data, MUON_KWARGS


class Fp32Muon(Muon):
    def zeropower_via_newtonschulz5(self, G, epsilon=1e-7):
        X = ops.cast(G, tf.float32)
        transposed = G.shape[-2] > G.shape[-1]
        if transposed:
            X = self.auto_transpose(X)
        X = X / (tf.norm(X, ord="fro", axis=[-2, -1], keepdims=True) + epsilon)
        for _ in range(self.ns_steps):
            A = ops.matmul(X, ops.transpose(X, [1, 0]))
            B = self.muon_b * A + self.muon_c * ops.matmul(A, A)
            X = self.muon_a * X + ops.matmul(B, X)
        return self.auto_transpose(X) if transposed else X


def time_one(label, opt_fn, train, test, epochs=2):
    tf.keras.utils.set_random_seed(7)
    m = build_resnet()
    m.compile(optimizer=opt_fn(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    m.fit(train, epochs=1, verbose=0)                      # warm up / compile
    t0 = time.time()
    h = m.fit(train, validation_data=test, epochs=epochs, verbose=0)
    print(f"  {label:<28} {(time.time()-t0)/epochs:6.1f}s/epoch   "
          f"val acc {max(h.history['val_accuracy']):.4f}", flush=True)


def main():
    """Steady state s/epoch for every arm at its own best lr.

    The sweep timings put XLA compilation inside the measured window, and Muon's graph
    (5 unrolled Newton-Schulz iterations per matrix) compiles slower than AdamW's, so
    those numbers overstate Muon's per epoch cost. This warms up for a full epoch first.
    """
    import json
    from exp_f_cifar import make_optimizer, ARMS
    best = json.load(open(os.path.join(ROOT, "tmp", "cifar_best_lr.json")))

    train, test = get_data()
    print(f"params {build_resnet().count_params():,}\n", flush=True)
    for arm in ARMS:
        lr = best[arm]
        time_one(f"{arm} @ {lr:g}", lambda a=arm, l=lr: make_optimizer(a, l), train, test, epochs=3)
    time_one("muon fp32 NS @ 0.1", lambda: Fp32Muon(learning_rate=0.1, **MUON_KWARGS), train, test, epochs=3)


if __name__ == "__main__":
    main()
