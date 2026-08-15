"""Experiment K: did fixing the nesterov blend help?

Re-runs ONLY muon and orthograd(muon) under the exact exp_j protocol (same seeds, epochs,
seeded data pipeline, lr, and last-2-epoch metric) so the results are directly comparable
to the exp_j numbers, which were produced with the OLD blend.

    old:  update = g + beta*m          (proportional to g + 0.95m)
    new:  update = (1-beta)*g + beta*m (proportional to g + 19m, matches torch lerp_)

Loop is seed-major, so stopping after any seed leaves a complete paired set.

    python tmp/exp_k_nesterov.py
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
from Net.Optimizers.ortho_grad import Orthograd
from exp_f_cifar import build_resnet, trunk_norm, MUON_KWARGS, BATCH
from exp_j_final import raw_cifar, aug, EPOCHS, SEEDS

ARMS = [("muon", 1e-1, lambda lr: Muon(learning_rate=lr, **MUON_KWARGS)),
        ("muon + ortho(grad)", 1e-1, lambda lr: Orthograd(Muon(learning_rate=lr, **MUON_KWARGS)))]

# exp_j results, identical protocol, OLD nesterov blend
OLD = {"muon": {7: 0.6883, 13: 0.6866, 29: 0.7005},
       "muon + ortho(grad)": {7: 0.6299, 13: 0.5525, 29: 0.6922}}


def main():
    (xtr, ytr), (xte, yte) = raw_cifar()
    base = tf.data.Dataset.from_tensor_slices((xtr, ytr))
    test = tf.data.Dataset.from_tensor_slices((xte, yte)).batch(512).prefetch(tf.data.AUTOTUNE)

    out, t0 = {}, time.time()
    for seed in SEEDS:
        for label, lr, make in ARMS:
            tf.keras.utils.set_random_seed(seed)
            train = (base.shuffle(10000, seed=seed, reshuffle_each_iteration=True)
                     .map(aug, num_parallel_calls=tf.data.AUTOTUNE)
                     .batch(BATCH).prefetch(tf.data.AUTOTUNE))
            model = build_resnet()
            w0 = trunk_norm(model)
            model.compile(optimizer=make(lr), loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])
            h = model.fit(train, validation_data=test, epochs=EPOCHS, verbose=0)
            va = h.history["val_accuracy"]
            stable = float(np.mean(va[-2:]))
            out.setdefault(label, {})[seed] = dict(
                stable=stable, best=float(max(va)),
                train=float(h.history["accuracy"][-1]), wgrowth=trunk_norm(model) / w0)
            print(f"  seed {seed}  {label:<20} last2 {stable:.4f}  "
                  f"(old {OLD[label][seed]:.4f}, delta {stable - OLD[label][seed]:+.4f})  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    print(f"\n{'arm':<22} {'NEW blend':>20} {'OLD blend (exp_j)':>20}")
    print("-" * 64)
    for label, _, _ in ARMS:
        done = [s for s in SEEDS if s in out.get(label, {})]
        n = np.array([out[label][s]["stable"] for s in done])
        o = np.array([OLD[label][s] for s in done])
        print(f"{label:<22} {n.mean():>11.4f} +/- {n.std():<6.4f} {o.mean():>11.4f} +/- {o.std():<6.4f}")

    print(f"\nPAIRED, new minus old, per seed")
    print(f"{'arm':<22}" + "".join(f"{f'seed {s}':>10}" for s in SEEDS) + f"{'mean':>10}{'sd':>9}")
    print("-" * 64)
    for label, _, _ in ARMS:
        done = [s for s in SEEDS if s in out.get(label, {})]
        d = np.array([out[label][s]["stable"] - OLD[label][s] for s in done])
        cells = "".join(f"{out[label][s]['stable'] - OLD[label][s]:>+10.4f}"
                        if s in out.get(label, {}) else f"{'-':>10}" for s in SEEDS)
        verdict = "BETTER" if d.mean() > 2 * d.std() else ("WORSE" if d.mean() < -2 * d.std() else "no effect")
        print(f"{label:<22}{cells}{d.mean():>+10.4f}{d.std():>9.4f}   {verdict}")

    print(f"\ntotal {time.time() - t0:.0f}s")
    with open(os.path.join(ROOT, "tmp", "results_nesterov.json"), "w") as f:
        json.dump({k: {str(s): v for s, v in d.items()} for k, d in out.items()}, f, indent=2)


if __name__ == "__main__":
    main()
