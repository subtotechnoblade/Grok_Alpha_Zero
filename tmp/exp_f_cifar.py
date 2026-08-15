"""Experiment F: optimizer benchmark on CIFAR-10 with a real ResNet.

The Connect4 self play data is a moving target (the labels come from a search that
is itself changing), so it is a poor benchmark for an optimizer. This uses a fixed
supervised dataset instead, but keeps YOUR ResNet_Block so the conv kernel shapes
and BatchNorm placement match the alpha zero net.

Arms: AdamW (control), Muon, BuggyMuon, Orthograd(Muon), Grokfast(Muon).

    python tmp/exp_f_cifar.py calibrate     # time one epoch, pick sizes
    python tmp/exp_f_cifar.py sweep         # short lr sweep, 1 seed
    python tmp/exp_f_cifar.py final         # best lr per arm, 3 seeds
"""
import os, sys, json, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import tensorflow as tf

for _dev in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_dev, True)

from Net.ResNet.ResNet_Block import ResNet_Block
from Net.Optimizers.muon import Muon
from Net.Optimizers.ortho_grad import Orthograd
from Net.Optimizers.grok_fast import Grokfast_EMA
from tmp.muon_buggy import BuggyMuon

BATCH = 256
SWEEP_EPOCHS = 5
FINAL_EPOCHS = 12
SEEDS = [7, 13, 29]


def build_resnet(filters=(32, 64, 128), blocks_per_stage=2, num_classes=10):
    """Same ResNet_Block you use for alpha zero, so kernels are (3, 3, C, C) and every
    conv feeds a BatchNorm - which is what makes the trunk scale invariant."""
    inputs = tf.keras.layers.Input((32, 32, 3), name="inputs")
    x = tf.keras.layers.Conv2D(filters[0], 3, padding="same",
                               kernel_initializer="he_normal", name="stem")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    for stage, f in enumerate(filters):
        # ResNet_Block only projects when the channel count changes, so stride and widen together
        x = ResNet_Block(f, (3, 3), strides=(1, 1) if stage == 0 else (2, 2), padding="same")(x)
        for _ in range(blocks_per_stage - 1):
            x = ResNet_Block(f, (3, 3), strides=(1, 1), padding="same")(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax",
                                    dtype="float32", name="head")(x)
    return tf.keras.Model(inputs, outputs)


def get_data():
    (xtr, ytr), (xte, yte) = tf.keras.datasets.cifar10.load_data()
    xtr, xte = xtr.astype("float32") / 255.0, xte.astype("float32") / 255.0
    mean, std = xtr.mean((0, 1, 2)), xtr.std((0, 1, 2))
    xtr, xte = (xtr - mean) / std, (xte - mean) / std
    ytr, yte = ytr.squeeze(), yte.squeeze()

    def aug(img, lab):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_crop(tf.pad(img, [[4, 4], [4, 4], [0, 0]], mode="REFLECT"), (32, 32, 3))
        return img, lab

    train = (tf.data.Dataset.from_tensor_slices((xtr, ytr)).shuffle(10000)
             .map(aug, num_parallel_calls=tf.data.AUTOTUNE)
             .batch(BATCH).prefetch(tf.data.AUTOTUNE))
    test = tf.data.Dataset.from_tensor_slices((xte, yte)).batch(512).prefetch(tf.data.AUTOTUNE)
    return train, test


MUON_KWARGS = dict(weight_decay=1e-4, use_nadam=False, adam_beta_1=0.9,
                   adam_beta_2=0.995, muon_beta=0.95,
                   exclude_layers=["stem", "head"])   # input and output layers -> AdamW


def make_optimizer(arm, lr):
    if arm == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4,
                                         beta_1=0.9, beta_2=0.995)
    if arm == "muon":
        return Muon(learning_rate=lr, **MUON_KWARGS)
    if arm == "muon_buggy":
        return BuggyMuon(learning_rate=lr, **MUON_KWARGS)
    if arm == "muon_ortho":
        return Orthograd(Muon(learning_rate=lr, **MUON_KWARGS))
    if arm == "muon_grokfast":
        return Grokfast_EMA(Muon(learning_rate=lr, **MUON_KWARGS), lamb=2.0)
    raise ValueError(arm)


ARMS = ["adamw", "muon", "muon_buggy", "muon_ortho", "muon_grokfast"]
SWEEP_LRS = {"adamw": [3e-4, 1e-3, 3e-3],
             # Muon's update has fixed norm rather than adaptive scale, so it wants a
             # higher lr than Adam. The buggy grid is shifted ~5x higher again because
             # its NS matrix has 3 rows, giving ||update|| = sqrt(3) instead of sqrt(C).
             "muon": [3e-3, 1e-2, 3e-2],
             "muon_buggy": [1e-2, 3e-2, 1e-1],
             "muon_ortho": [3e-3, 1e-2, 3e-2],
             "muon_grokfast": [3e-3, 1e-2, 3e-2]}

# The first sweep peaked at a grid edge for 4 of 5 arms, so the optima were outside it.
# These extend each grid in the direction it was still improving.
EXTEND_LRS = {"adamw": [1e-2, 3e-2],
              "muon": [1e-1],
              "muon_buggy": [1e-3, 3e-3],     # buggy got WORSE as lr rose, so extend down
              "muon_ortho": [1e-1],
              "muon_grokfast": [1e-1]}

# After sweep2: adamw peaks at 3e-3, buggy at 1e-2, grokfast at 3e-2 - all bracketed.
# muon and muon_ortho were still rising at 1e-1, so they need one more point each.
EXTEND2_LRS = {"adamw": [], "muon": [3e-1], "muon_buggy": [],
               "muon_ortho": [3e-1], "muon_grokfast": []}


def trunk_norm(model):
    ws = [v for v in model.trainable_variables if len(v.shape) == 4 and v.shape[0] == 3]
    return float(np.mean([float(tf.norm(v)) for v in ws]))


def run(arm, lr, seed, train, test, epochs):
    tf.keras.utils.set_random_seed(seed)
    model = build_resnet()
    w0 = trunk_norm(model)
    model.compile(optimizer=make_optimizer(arm, lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    t0 = time.time()
    h = model.fit(train, validation_data=test, epochs=epochs, verbose=0)
    return dict(best_acc=float(max(h.history["val_accuracy"])),
                final_acc=float(h.history["val_accuracy"][-1]),
                final_train_acc=float(h.history["accuracy"][-1]),
                wgrowth=trunk_norm(model) / w0,
                secs=time.time() - t0,
                curve=[float(a) for a in h.history["val_accuracy"]])


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "calibrate"
    train, test = get_data()
    print(f"model params: {build_resnet().count_params():,}   mode: {mode}", flush=True)

    if mode == "calibrate":
        for arm in ARMS:
            r = run(arm, SWEEP_LRS[arm][1], SEEDS[0], train, test, 1)
            print(f"  {arm:<16} {r['secs']:6.1f}s/epoch   val acc after 1 epoch {r['best_acc']:.4f}", flush=True)
        return

    results = {}
    if mode in ("sweep", "sweep2", "sweep3"):
        grid = {"sweep": SWEEP_LRS, "sweep2": EXTEND_LRS, "sweep3": EXTEND2_LRS}[mode]
        for arm in ARMS:
            for lr in grid[arm]:
                r = run(arm, lr, SEEDS[0], train, test, SWEEP_EPOCHS)
                results[f"{arm}@{lr:g}"] = r
                print(f"  {arm:<16} lr {lr:<8g} best acc {r['best_acc']:.4f}  "
                      f"train acc {r['final_train_acc']:.4f}  |W| x{r['wgrowth']:.3f}  "
                      f"{r['secs']:.0f}s", flush=True)
    elif mode == "final":
        best = json.load(open(os.path.join(ROOT, "tmp", "cifar_best_lr.json")))
        for arm in ARMS:
            for seed in SEEDS:
                r = run(arm, best[arm], seed, train, test, FINAL_EPOCHS)
                results.setdefault(arm, []).append(r)
                print(f"  {arm:<16} seed {seed} lr {best[arm]:<8g} best acc {r['best_acc']:.4f}  "
                      f"|W| x{r['wgrowth']:.3f}  {r['secs']:.0f}s", flush=True)
        print(f"\n{'arm':<16} {'best test acc':>22} {'train acc':>12} {'|W| growth':>12}")
        print("-" * 64)
        for arm in ARMS:
            a = np.array([r["best_acc"] for r in results[arm]])
            t = np.array([r["final_train_acc"] for r in results[arm]])
            w = np.array([r["wgrowth"] for r in results[arm]])
            print(f"{arm:<16} {a.mean():>11.4f} +/- {a.std():<7.4f} {t.mean():>11.4f} {w.mean():>11.2f}x")

    with open(os.path.join(ROOT, "tmp", f"results_cifar_{mode}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote tmp/results_cifar_{mode}.json")


if __name__ == "__main__":
    main()
