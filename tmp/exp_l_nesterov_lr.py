"""Experiment L: lr sweep for the CORRECTED nesterov blend.

exp_k showed the reference blend loses at lr=1e-1 (0.6110 vs 0.6918) but with an 11x
variance blow-up (sd 0.0062 -> 0.0676) and a collapse on one seed - an instability
signature, i.e. the lr was tuned for the old blend and is now too high.

Note the (1-beta) factor does NOT shrink the step: newton-schulz divides by the Frobenius
norm first, so any constant scaling of its input is erased. The applied update norm is
identical for both blends. What changes is DIRECTION - the new blend is momentum
dominated, so consecutive steps are more correlated, giving more net displacement per
step and therefore wanting a lower nominal lr.

Same exp_j protocol throughout. Seed-major so partial runs are usable.

    python tmp/exp_l_nesterov_lr.py
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
from exp_f_cifar import build_resnet, trunk_norm, MUON_KWARGS, BATCH
from exp_j_final import raw_cifar, aug, EPOCHS, SEEDS

LRS = [1e-3, 5e-3, 1e-2]
TARGET = 0.6918          # muon, OLD blend, lr 1e-1, same protocol (exp_j)
OLD_PER_SEED = {7: 0.6883, 13: 0.6866, 29: 0.7005}
NEW_AT_1E1 = {7: 0.6389, 13: 0.6762, 29: 0.5178}     # exp_k


def main():
    (xtr, ytr), (xte, yte) = raw_cifar()
    base = tf.data.Dataset.from_tensor_slices((xtr, ytr))
    test = tf.data.Dataset.from_tensor_slices((xte, yte)).batch(512).prefetch(tf.data.AUTOTUNE)

    out, t0 = {}, time.time()
    for seed in SEEDS:
        for lr in LRS:
            tf.keras.utils.set_random_seed(seed)
            train = (base.shuffle(10000, seed=seed, reshuffle_each_iteration=True)
                     .map(aug, num_parallel_calls=tf.data.AUTOTUNE)
                     .batch(BATCH).prefetch(tf.data.AUTOTUNE))
            model = build_resnet()
            w0 = trunk_norm(model)
            model.compile(optimizer=Muon(learning_rate=lr, **MUON_KWARGS),
                          loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            h = model.fit(train, validation_data=test, epochs=EPOCHS, verbose=0)
            va = h.history["val_accuracy"]
            stable = float(np.mean(va[-2:]))
            out.setdefault(lr, {})[seed] = dict(
                stable=stable, best=float(max(va)),
                train=float(h.history["accuracy"][-1]), wgrowth=trunk_norm(model) / w0)
            print(f"  seed {seed}  lr {lr:<7g} last2 {stable:.4f}  "
                  f"(old blend @1e-1: {OLD_PER_SEED[seed]:.4f}, delta {stable - OLD_PER_SEED[seed]:+.4f})  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    print(f"\n{'config':<34} {'test acc (last 2 ep)':>22} {'train acc':>11} {'|W|':>8}")
    print("-" * 78)
    for lr in LRS:
        done = [s for s in SEEDS if s in out.get(lr, {})]
        if not done:
            continue
        a = np.array([out[lr][s]["stable"] for s in done])
        t = np.array([out[lr][s]["train"] for s in done])
        w = np.array([out[lr][s]["wgrowth"] for s in done])
        flag = "  <- beats old blend" if a.mean() > TARGET else ""
        print(f"{'NEW blend @ ' + f'{lr:g}':<34} {a.mean():>11.4f} +/- {a.std():<7.4f} "
              f"{t.mean():>10.4f} {w.mean():>7.2f}x{flag}")
    n1 = np.array(list(NEW_AT_1E1.values()))
    o1 = np.array(list(OLD_PER_SEED.values()))
    print(f"{'NEW blend @ 0.1 (exp_k)':<34} {n1.mean():>11.4f} +/- {n1.std():<7.4f}")
    print(f"{'OLD blend @ 0.1 (exp_j, the target)':<34} {o1.mean():>11.4f} +/- {o1.std():<7.4f}")

    print(f"\ntotal {time.time() - t0:.0f}s")
    with open(os.path.join(ROOT, "tmp", "results_nesterov_lr.json"), "w") as f:
        json.dump({str(k): {str(s): v for s, v in d.items()} for k, d in out.items()}, f, indent=2)


if __name__ == "__main__":
    main()
