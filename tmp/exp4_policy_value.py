"""EXP 4 - policy and value on the SAME epoch axis, and where the value overfit lives.

Three questions, one run:

  Q1  how good is the policy at epoch 1-2?  -> held-out policy CE / KL / top-1 per
      epoch, next to the value wMSE, same split as gbt_target.py. If held-out policy
      keeps improving to epoch 8+ while value turns at 2, the conflict is real. If
      held-out policy ALSO turns at 2-3, "the policy needs 8 epochs" is a train-loss
      artefact and there is nothing to fix.

  Q2  can the value be slowed instead of stopped?  -> value-head LR /5 and /25, and
      value LOSS weight 0.2. Note these are NOT the same knob: Adam is scale
      invariant, so a loss weight leaves the value head's own updates unchanged and
      only changes the TRUNK's gradient mix. LR scaling is the head knob.

  Q3  is the overfit in the value HEAD or in the shared TRUNK?  -> at epochs 2/6/12
      of the baseline, freeze the trunk, re-init the value head, retrain it alone.
      Trunk fine + head bad  => early-stopping/slowing the head can work.
      Trunk degrades         => no head-side trick saves it, the representation
                                itself is memorising and the fix must be upstream.

  GBT raw 0.6126 | GBT hand+rel 0.5681 | const-0 0.7691
"""
import os
import re
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.chdir(ROOT)

import numpy as np
import tensorflow as tf

from exp2_optimizer import load_split, dedup, wmse

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
PROBE_AT = (2, 6, 12)
PROBE_EPOCHS = 4
BATCH = 1024
LR = 3e-4
VALUE_VARS = r"^(vh_|value_1)"


def make_model(value_filters=2, value_dense=32):
    """Mirrors Connect4/Build_Model.build_model. Every layer named so weights can be
    copied between two independently constructed models."""
    from Net.ResNet.ResNet_Block import ResNet_Block
    from Connect4.config import build_config

    nf = build_config["num_filters"]
    inputs = tf.keras.layers.Input(batch_shape=(None, 6, 7, 4), name="inputs")
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", use_bias=False,
                               kernel_initializer="he_normal", name="eyes")(inputs)
    x = tf.keras.layers.BatchNormalization(name="eyes_bn")(x)
    x = tf.keras.layers.Activation("relu", name="eyes_relu")(x)
    for i in range(build_config["num_resnet_layers"]):
        x = ResNet_Block(nf, (3, 3), strides=(1, 1), padding="same", name=f"res_{i}")(x)

    p = tf.keras.layers.BatchNormalization(name="ph_bn0")(x)
    p = tf.keras.layers.Activation("relu", name="ph_relu0")(p)
    p = tf.keras.layers.Conv2D(8, (3, 3), padding="same", use_bias=False,
                               kernel_initializer="he_normal", name="ph_conv")(p)
    p = tf.keras.layers.BatchNormalization(name="ph_bn1")(p)
    p = tf.keras.layers.Activation("relu", name="ph_relu1")(p)
    p = tf.keras.layers.Reshape((p.shape[-3] * p.shape[-2] * p.shape[-1],), name="ph_flat")(p)
    p = tf.keras.layers.Dense(128, use_bias=False, kernel_initializer="he_normal", name="ph_d0")(p)
    p = tf.keras.layers.BatchNormalization(name="ph_bn2")(p)
    p = tf.keras.layers.Activation("relu", name="ph_relu2")(p)
    p = tf.keras.layers.Dense(64, use_bias=False, kernel_initializer="he_normal", name="ph_d1")(p)
    p = tf.keras.layers.BatchNormalization(name="ph_bn3")(p)
    p = tf.keras.layers.Activation("relu", name="ph_relu3")(p)
    p = tf.keras.layers.Dense(7, dtype="float32", name="policy_1")(p)
    p = tf.keras.layers.Activation("softmax", dtype="float32", name="policy")(p)

    v = tf.keras.layers.BatchNormalization(name="vh_bn0")(x)
    v = tf.keras.layers.Activation("relu", name="vh_relu0")(v)
    v = tf.keras.layers.Conv2D(value_filters, (1, 1), padding="same", use_bias=False,
                               kernel_initializer="he_normal", name="vh_conv")(v)
    v = tf.keras.layers.BatchNormalization(name="vh_bn1")(v)
    v = tf.keras.layers.Activation("relu", name="vh_relu1")(v)
    v = tf.keras.layers.Reshape((v.shape[-3] * v.shape[-2] * v.shape[-1],), name="vh_flat")(v)
    v = tf.keras.layers.Dense(value_dense, use_bias=False, kernel_initializer="he_normal",
                              name="vh_d0")(v)
    v = tf.keras.layers.BatchNormalization(name="vh_bn2")(v)
    v = tf.keras.layers.Activation("relu", name="vh_relu2")(v)
    v = tf.keras.layers.Dense(1, dtype="float32", name="value_1")(v)
    v = tf.keras.layers.Activation("tanh", dtype="float32", name="value")(v)
    return tf.keras.Model(inputs=inputs, outputs=[p, v])


def scaled_adam(patterns=(), scale=1.0, **kw):
    """AdamW with a per-variable learning-rate multiplier for matching paths."""
    from Net.Optimizers.adam import Adam

    class _LRScaled(Adam):
        def build(self, variables):
            super().build(variables)
            self._mult = [scale if any(re.search(p, v.path) for p in patterns) else 1.0
                          for v in variables]

        def update_step(self, gradient, variable, learning_rate):
            super().update_step(gradient, variable,
                                learning_rate * self._mult[self._get_variable_index(variable)])

    return _LRScaled(**kw)


def compiled(value_lr_scale=1.0, value_loss_weight=1.0, value_filters=2):
    from Net.Alpha_Loss import Policy_Loss, Value_Loss
    from Net.Optimizers.adam import Adam
    m = make_model(value_filters)
    if value_lr_scale == 1.0:
        opt = Adam(learning_rate=LR, weight_decay=0.004, beta_1=0.9, beta_2=0.995)
    else:
        opt = scaled_adam([VALUE_VARS], value_lr_scale, learning_rate=LR,
                          weight_decay=0.004, beta_1=0.9, beta_2=0.995)
    m.compile(optimizer=opt,
              loss={"policy": Policy_Loss(dtype="float32"),
                    "value": Value_Loss(dtype="float32")},
              loss_weights={"policy": 1.0, "value": value_loss_weight})
    return m


class Scorer:
    """Held-out value wMSE (+ ply buckets) and policy CE / KL / top-1, all occurrence
    weighted so every number stays comparable to the GBT bars."""

    def __init__(self, Xte, Pte, yte, wte, ply):
        self.X, self.P, self.y, self.w = Xte, Pte, yte, wte
        self.m = (ply <= 9, (ply >= 10) & (ply <= 21), ply >= 22)
        q = np.clip(Pte, 1e-12, 1.0)
        self.H = float(np.sum(wte * -(Pte * np.log(q)).sum(1)) / wte.sum())
        self.argmax = Pte.argmax(1)
        self.v0 = wmse(yte, np.zeros(len(yte)), wte)

    def __call__(self, model):
        pp, pv = model.predict(self.X, batch_size=4096, verbose=0)
        pv = pv[:, 0].astype(np.float64)
        pp = np.clip(pp.astype(np.float64), 1e-12, 1.0)
        ce = float(np.sum(self.w * -(self.P * np.log(pp)).sum(1)) / self.w.sum())
        top1 = float(np.sum(self.w * (pp.argmax(1) == self.argmax)) / self.w.sum())
        return dict(v=wmse(self.y, pv, self.w),
                    vb=[wmse(self.y[m], pv[m], self.w[m]) for m in self.m],
                    ce=ce, kl=ce - self.H, top1=top1, std=float(pv.std()))


HDR = (f"{'arm':<20} {'ep':>3} {'VALUE':>8} {'vs c0':>7} {'ply22+':>8} | "
       f"{'polCE':>7} {'polKL':>7} {'top1':>7}")


def row(tag, ep, r, v0):
    return (f"{tag:<20} {ep:>3} {r['v']:>8.4f} {1-r['v']/v0:>6.1%} {r['vb'][2]:>8.4f} | "
            f"{r['ce']:>7.4f} {r['kl']:>7.4f} {r['top1']:>6.1%}")


def probe_trunk(trained, Xtr, Vtr, w, score, tag):
    """Freeze everything but the value head, re-init the head, retrain it alone.
    Isolates trunk quality from head quality at this point in training."""
    from Net.Alpha_Loss import Policy_Loss, Value_Loss
    from Net.Optimizers.adam import Adam
    tf.keras.utils.set_random_seed(1)
    fresh = make_model()
    for layer in fresh.layers:
        if not (layer.name.startswith("vh_") or layer.name == "value_1"):
            layer.set_weights(trained.get_layer(layer.name).get_weights())
            layer.trainable = False
    fresh.compile(optimizer=Adam(learning_rate=LR, weight_decay=0.004,
                                 beta_1=0.9, beta_2=0.995),
                  loss={"policy": Policy_Loss(dtype="float32"),
                        "value": Value_Loss(dtype="float32")},
                  loss_weights={"policy": 0.0, "value": 1.0})
    best = 1e9
    for ep in range(1, PROBE_EPOCHS + 1):
        fresh.fit(Xtr, (np.zeros((len(Xtr), 7), np.float32), Vtr), sample_weight=(w, w),
                  batch_size=BATCH, epochs=1, verbose=0, shuffle=True)
        best = min(best, score(fresh)["v"])
    print(f"    [probe] frozen trunk @ep{tag}, fresh head, {PROBE_EPOCHS} ep -> "
          f"best value {best:.4f}", flush=True)
    del fresh


def main():
    d = load_split()
    trX, trP, trY, trW = dedup(*d["tr"])
    teX, teP, teY, teW = dedup(*d["te"])
    ply = (teX[..., -1] != 0).reshape(len(teX), -1).sum(1)
    score = Scorer(teX.astype(np.float32), teP, teY, teW, ply)
    v0 = score.v0

    Xtr, Ptr = trX.astype(np.float32), trP.astype(np.float32)
    Vtr = trY.astype(np.float32)[:, None]
    w = trW.astype(np.float32)
    steps = int(np.ceil(len(trX) / BATCH))

    print(f"train {len(trX)} unique | {steps} steps/ep | held-out {len(teX)}")
    print(f"value: const-0 {v0:.4f} | GBT raw 0.6126")
    print(f"policy: target entropy floor {score.H:.4f} (polKL = CE - floor); "
          f"uniform-over-7 CE {np.log(7):.4f}\n")

    tf.keras.utils.set_random_seed(0)
    r = score(compiled())
    print(HDR)
    print("-" * len(HDR))
    print(row("UNTRAINED", 0, r, v0) + "\n")

    arms = [("baseline", dict()),
            ("value lr /5", dict(value_lr_scale=0.2)),
            ("value lr /25", dict(value_lr_scale=0.04)),
            ("value loss w 0.2", dict(value_loss_weight=0.2))]

    for tag, kw in arms:
        tf.keras.utils.set_random_seed(0)
        model = compiled(**kw)
        bv, bp = (1e9, -1), (1e9, -1)
        for ep in range(1, EPOCHS + 1):
            model.fit(Xtr, (Ptr, Vtr), sample_weight=(w, w), batch_size=BATCH,
                      epochs=1, verbose=0, shuffle=True)
            r = score(model)
            bv = min(bv, (r["v"], ep))
            bp = min(bp, (r["kl"], ep))
            print(row(tag if ep == 1 else "", ep, r, v0), flush=True)
            if tag == "baseline" and ep in PROBE_AT:
                probe_trunk(model, Xtr, Vtr, w, score, ep)
        print(f"{'  BEST':<20} value {bv[0]:.4f} @ep{bv[1]}   "
              f"policyKL {bp[0]:.4f} @ep{bp[1]}\n", flush=True)
        del model
        tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()
