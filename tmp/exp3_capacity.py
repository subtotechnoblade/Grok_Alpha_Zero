"""EXP 3 - untrained baseline + value-head capacity, same split as gbt_target.py.

Row 0 is the UNTRAINED network: random init, zero gradient steps. It is the control
for "can epoch 1-2 possibly be good" - if epoch 1 already beats it by a wide margin,
2 passes over the data is doing real work.

Then value-head filters 1 / 2 / 4 / 16 (2 is current), plus dropout on the current
head. Everything else identical: same trunk, same policy head, AdamW wd 0.004,
lr 3e-4, batch 1024, seed 0.

  GBT raw 0.6126 | GBT hand+rel 0.5681 | const-0 0.7691
"""
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import tensorflow as tf

from exp2_optimizer import load_split, dedup, wmse

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
BATCH = 1024


def make_model(value_filters=2, value_dropout=0.0):
    """Mirrors Connect4/Build_Model.build_model; only the value head is parameterised."""
    from Net.ResNet.ResNet_Block import ResNet_Block
    from Connect4.config import build_config

    nf = build_config["num_filters"]
    inputs = tf.keras.layers.Input(batch_shape=(None, 6, 7, 4), name="inputs")
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", use_bias=False,
                               kernel_initializer="he_normal", name="eyes")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    for _ in range(build_config["num_resnet_layers"]):
        x = ResNet_Block(nf, (3, 3), strides=(1, 1), padding="same")(x)

    p = tf.keras.layers.Activation("relu")(tf.keras.layers.BatchNormalization()(x))
    p = tf.keras.layers.Conv2D(8, (3, 3), padding="same", use_bias=False,
                               kernel_initializer="he_normal")(p)
    p = tf.keras.layers.Activation("relu")(tf.keras.layers.BatchNormalization()(p))
    p = tf.keras.layers.Reshape((p.shape[-3] * p.shape[-2] * p.shape[-1],))(p)
    p = tf.keras.layers.Dense(128, use_bias=False, kernel_initializer="he_normal")(p)
    p = tf.keras.layers.Activation("relu")(tf.keras.layers.BatchNormalization()(p))
    p = tf.keras.layers.Dense(64, use_bias=False, kernel_initializer="he_normal")(p)
    p = tf.keras.layers.Activation("relu")(tf.keras.layers.BatchNormalization()(p))
    p = tf.keras.layers.Dense(7, dtype="float32", name="policy_1")(p)
    p = tf.keras.layers.Activation("softmax", dtype="float32", name="policy")(p)

    v = tf.keras.layers.Activation("relu")(tf.keras.layers.BatchNormalization()(x))
    v = tf.keras.layers.Conv2D(value_filters, (1, 1), padding="same", use_bias=False,
                               kernel_initializer="he_normal")(v)
    v = tf.keras.layers.Activation("relu")(tf.keras.layers.BatchNormalization()(v))
    v = tf.keras.layers.Reshape((v.shape[-3] * v.shape[-2] * v.shape[-1],))(v)
    if value_dropout:
        v = tf.keras.layers.Dropout(value_dropout)(v)
    v = tf.keras.layers.Dense(32, use_bias=False, kernel_initializer="he_normal")(v)
    v = tf.keras.layers.Activation("relu")(tf.keras.layers.BatchNormalization()(v))
    v = tf.keras.layers.Dense(1, dtype="float32", name="value_1")(v)
    v = tf.keras.layers.Activation("tanh", dtype="float32", name="value")(v)
    return tf.keras.Model(inputs=inputs, outputs=[p, v])


def compiled(**kw):
    from Net.Alpha_Loss import Policy_Loss, Value_Loss
    from Net.Optimizers.adam import Adam
    m = make_model(**kw)
    m.compile(optimizer=Adam(learning_rate=3e-4, weight_decay=0.004,
                             beta_1=0.9, beta_2=0.995),
              loss={"policy": Policy_Loss(dtype="float32"),
                    "value": Value_Loss(dtype="float32")})
    return m


def main():
    d = load_split()
    trX, trP, trY, trW = dedup(*d["tr"])
    teX, teP, teY, teW = dedup(*d["te"])
    ply = (teX[..., -1] != 0).reshape(len(teX), -1).sum(1)
    base = wmse(teY, np.zeros(len(teY)), teW)
    Xtr, Ptr = trX.astype(np.float32), trP.astype(np.float32)
    Vtr = trY.astype(np.float32)[:, None]
    Xte = teX.astype(np.float32)
    w = trW.astype(np.float32)
    m1, m2, m3 = ply <= 9, (ply >= 10) & (ply <= 21), ply >= 22
    steps = int(np.ceil(len(trX) / BATCH))

    def score(model):
        pv = model.predict(Xte, batch_size=4096, verbose=0)[1][:, 0].astype(np.float64)
        return (wmse(teY, pv, teW), wmse(teY[m1], pv[m1], teW[m1]),
                wmse(teY[m2], pv[m2], teW[m2]), wmse(teY[m3], pv[m3], teW[m3]),
                float(pv.std()))

    print(f"train {len(trX)} unique | {steps} steps/epoch | held-out {len(teX)}")
    print(f"const-0 {base:.4f} | GBT raw 0.6126 | GBT hand+rel 0.5681\n")
    print(f"{'arm':<22} {'ep':>3} {'passes':>7} {'HELD-OUT':>9} {'vs c0':>7} "
          f"{'ply0-9':>8} {'ply10-21':>9} {'ply22+':>8} {'predstd':>8}")
    print("-" * 92)

    tf.keras.utils.set_random_seed(0)
    r = score(compiled(value_filters=2))
    print(f"{'UNTRAINED (random)':<22} {0:>3} {0.0:>7.2f} {r[0]:>9.4f} "
          f"{1-r[0]/base:>6.1%} {r[1]:>8.4f} {r[2]:>9.4f} {r[3]:>8.4f} {r[4]:>8.3f}\n")

    arms = [("filters=1", dict(value_filters=1)),
            ("filters=2 (current)", dict(value_filters=2)),
            ("filters=4", dict(value_filters=4)),
            ("filters=16", dict(value_filters=16)),
            ("filters=2 drop0.3", dict(value_filters=2, value_dropout=0.3))]
    for tag, kw in arms:
        tf.keras.utils.set_random_seed(0)
        model = compiled(**kw)
        best = (1e9, -1, None)
        for ep in range(1, EPOCHS + 1):
            model.fit(Xtr, (Ptr, Vtr), sample_weight=(w, w), batch_size=BATCH,
                      epochs=1, verbose=0, shuffle=True)
            r = score(model)
            if r[0] < best[0]:
                best = (r[0], ep, r)
            print(f"{tag if ep == 1 else '':<22} {ep:>3} {ep*steps*BATCH/len(trX):>7.2f} "
                  f"{r[0]:>9.4f} {1-r[0]/base:>6.1%} {r[1]:>8.4f} {r[2]:>9.4f} "
                  f"{r[3]:>8.4f} {r[4]:>8.3f}", flush=True)
        b = best[2]
        print(f"{'  BEST':<22} {best[1]:>3} {best[1]*steps*BATCH/len(trX):>7.2f} "
              f"{b[0]:>9.4f} {1-b[0]/base:>6.1%} {b[1]:>8.4f} {b[2]:>9.4f} "
              f"{b[3]:>8.4f} {b[4]:>8.3f}\n", flush=True)


if __name__ == "__main__":
    main()
