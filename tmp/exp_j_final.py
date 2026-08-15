"""Experiment J: the definitive orthograd test. With or without, once and for all.

Fixes the three methodology faults that made the earlier single-seed numbers unreliable
(same config scored 0.7728 in one script and 0.6920 in another):

  1. SEEDED DATA ORDER. The dataset pipeline is rebuilt per run after set_random_seed,
     so a given seed gives every arm the identical shuffle and identical augmentation
     stream. Previously each script invocation saw a different data order.
  2. STABLE METRIC. Reports the mean of the last 2 epochs' val accuracy instead of
     max-over-epochs, which on an unconverged model is mostly noise.
  3. PAIRED ANALYSIS. Every arm runs on every seed, so ortho-vs-base is measured as a
     within-seed difference. That cancels the init and data-order variance that swamped
     the unpaired comparison.

Each arm uses the best lr found in the earlier sweeps for its family.

    python tmp/exp_j_final.py
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
from exp_h_ortho import WrapSafeAdam
from post_ortho import PostOrthograd

EPOCHS = 6
SEEDS = [7, 13, 29]
PAIRS = [
    ("adam", 3e-3, lambda lr: WrapSafeAdam(lr, wd=0.0)),
    ("adam + ortho(grad)", 3e-3, lambda lr: Orthograd(WrapSafeAdam(lr, wd=0.0))),
    ("adam + ortho(update)", 3e-3, lambda lr: PostOrthograd(WrapSafeAdam(lr, wd=0.0))),
    ("muon", 1e-1, lambda lr: Muon(learning_rate=lr, **MUON_KWARGS)),
    ("muon + ortho(grad)", 1e-1, lambda lr: Orthograd(Muon(learning_rate=lr, **MUON_KWARGS))),
    ("muon + ortho(update)", 1e-1, lambda lr: PostOrthograd(Muon(learning_rate=lr, **MUON_KWARGS))),
]
BASE_OF = {"adam + ortho(grad)": "adam", "adam + ortho(update)": "adam",
           "muon + ortho(grad)": "muon", "muon + ortho(update)": "muon"}


def raw_cifar():
    (xtr, ytr), (xte, yte) = tf.keras.datasets.cifar10.load_data()
    xtr, xte = xtr.astype("float32") / 255.0, xte.astype("float32") / 255.0
    mean, std = xtr.mean((0, 1, 2)), xtr.std((0, 1, 2))
    return ((xtr - mean) / std, ytr.squeeze()), ((xte - mean) / std, yte.squeeze())


def aug(img, lab):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_crop(tf.pad(img, [[4, 4], [4, 4], [0, 0]], mode="REFLECT"), (32, 32, 3))
    return img, lab


def main():
    (xtr, ytr), (xte, yte) = raw_cifar()
    base = tf.data.Dataset.from_tensor_slices((xtr, ytr))
    test = tf.data.Dataset.from_tensor_slices((xte, yte)).batch(512).prefetch(tf.data.AUTOTUNE)

    out, t0 = {}, time.time()
    for seed in SEEDS:
        for label, lr, make in PAIRS:
            tf.keras.utils.set_random_seed(seed)          # seeds init AND the aug stream
            train = (base.shuffle(10000, seed=seed, reshuffle_each_iteration=True)
                     .map(aug, num_parallel_calls=tf.data.AUTOTUNE)
                     .batch(BATCH).prefetch(tf.data.AUTOTUNE))
            model = build_resnet()
            w0 = trunk_norm(model)
            model.compile(optimizer=make(lr), loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])
            h = model.fit(train, validation_data=test, epochs=EPOCHS, verbose=0)
            va = h.history["val_accuracy"]
            r = dict(stable=float(np.mean(va[-2:])), best=float(max(va)),
                     train=float(h.history["accuracy"][-1]),
                     wgrowth=trunk_norm(model) / w0)
            out.setdefault(label, {})[seed] = r
            print(f"  seed {seed}  {label:<22} last2 {r['stable']:.4f}  best {r['best']:.4f}  "
                  f"|W| x{r['wgrowth']:.2f}  [{time.time()-t0:.0f}s]", flush=True)

    print(f"\n{'arm':<24} {'test acc (last 2 ep)':>24} {'train acc':>11} {'|W| growth':>12}")
    print("-" * 74)
    for label, _, _ in PAIRS:
        s = np.array([out[label][k]["stable"] for k in SEEDS])
        t = np.array([out[label][k]["train"] for k in SEEDS])
        w = np.array([out[label][k]["wgrowth"] for k in SEEDS])
        print(f"{label:<24} {s.mean():>13.4f} +/- {s.std():<7.4f} {t.mean():>10.4f} {w.mean():>11.2f}x")

    print(f"\nPAIRED differences, ortho minus its own baseline, per seed")
    print(f"{'comparison':<24}" + "".join(f"{f'seed {k}':>10}" for k in SEEDS) + f"{'mean':>10}{'sd':>9}")
    print("-" * 74)
    for label in [p[0] for p in PAIRS if p[0] in BASE_OF]:
        d = np.array([out[label][k]["stable"] - out[BASE_OF[label]][k]["stable"] for k in SEEDS])
        verdict = "HELPS" if d.mean() > 2 * d.std() else ("HURTS" if d.mean() < -2 * d.std() else "no effect")
        print(f"{label:<24}" + "".join(f"{x:>+10.4f}" for x in d) +
              f"{d.mean():>+10.4f}{d.std():>9.4f}   {verdict}")

    print(f"\ntotal {time.time() - t0:.0f}s")
    with open(os.path.join(ROOT, "tmp", "results_final_ortho.json"), "w") as f:
        json.dump({k: {str(s): v for s, v in d.items()} for k, d in out.items()}, f, indent=2)


if __name__ == "__main__":
    main()
