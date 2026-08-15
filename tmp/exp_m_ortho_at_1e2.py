"""Experiment M: orthograd(muon) at lr 1e-2 with the corrected nesterov blend.

3 runs. The muon baseline at the same lr / blend / protocol already exists (exp_l), so
only the ortho arm runs. Paired against it per seed.

    python tmp/exp_m_ortho_at_1e2.py
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

LR = 1e-2
# exp_l, corrected blend, lr 1e-2, identical protocol
MUON = {7: 0.7494, 13: 0.7377, 29: 0.6942}


def main():
    (xtr, ytr), (xte, yte) = raw_cifar()
    base = tf.data.Dataset.from_tensor_slices((xtr, ytr))
    test = tf.data.Dataset.from_tensor_slices((xte, yte)).batch(512).prefetch(tf.data.AUTOTUNE)

    out, t0 = {}, time.time()
    for seed in SEEDS:
        tf.keras.utils.set_random_seed(seed)
        train = (base.shuffle(10000, seed=seed, reshuffle_each_iteration=True)
                 .map(aug, num_parallel_calls=tf.data.AUTOTUNE)
                 .batch(BATCH).prefetch(tf.data.AUTOTUNE))
        model = build_resnet()
        w0 = trunk_norm(model)
        model.compile(optimizer=Orthograd(Muon(learning_rate=LR, **MUON_KWARGS)),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        h = model.fit(train, validation_data=test, epochs=EPOCHS, verbose=0)
        va = h.history["val_accuracy"]
        stable = float(np.mean(va[-2:]))
        out[seed] = dict(stable=stable, best=float(max(va)),
                         train=float(h.history["accuracy"][-1]),
                         wgrowth=trunk_norm(model) / w0)
        print(f"  seed {seed}  ortho(muon) last2 {stable:.4f}  "
              f"(muon {MUON[seed]:.4f}, delta {stable - MUON[seed]:+.4f})  "
              f"|W| x{out[seed]['wgrowth']:.2f}  [{time.time()-t0:.0f}s]", flush=True)

    o = np.array([out[s]["stable"] for s in SEEDS])
    m = np.array([MUON[s] for s in SEEDS])
    d = o - m
    print(f"\n{'config':<28} {'test acc (last 2 ep)':>24} {'|W|':>8}")
    print("-" * 62)
    print(f"{'muon @ 1e-2 (exp_l)':<28} {m.mean():>13.4f} +/- {m.std():<7.4f} {'1.04x':>8}")
    print(f"{'ortho(muon) @ 1e-2':<28} {o.mean():>13.4f} +/- {o.std():<7.4f} "
          f"{np.mean([out[s]['wgrowth'] for s in SEEDS]):>7.2f}x")
    verdict = "HELPS" if d.mean() > 2 * d.std() else ("HURTS" if d.mean() < -2 * d.std() else "no effect")
    print(f"\nPAIRED, ortho minus muon: " + "  ".join(f"seed {s} {d[i]:+.4f}" for i, s in enumerate(SEEDS)))
    print(f"mean {d.mean():+.4f}  sd {d.std():.4f}   -> {verdict}")
    print(f"\ntotal {time.time() - t0:.0f}s")

    with open(os.path.join(ROOT, "tmp", "results_ortho_1e2.json"), "w") as f:
        json.dump({str(s): v for s, v in out.items()}, f, indent=2)


if __name__ == "__main__":
    main()
