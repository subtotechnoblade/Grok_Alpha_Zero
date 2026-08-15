"""Stage 5 verification: is the per-sample dedup weight actually applied?

Value_Loss.call / Policy_Loss.call invoke an INSTANCE of
tf.keras.losses.MeanSquaredError / CategoricalCrossentropy. Calling an instance
runs its full __call__, which REDUCES over the batch to a scalar. Keras then
applies the sample_weight to whatever call() returned. If that is already a
scalar, the weights cannot weight anything per-sample: the scalar broadcasts
against the weight vector and the mean of that is scalar * mean(batch weights).

Hypotheses for the reported value loss on a batch with weights w_i:
  H_per_sample : sum_i(w_i * se_i) / N        (what the dedup math assumes)
  H_broadcast  : (sum_i(se_i) / N) * mean(w)  (weights degrade to a scalar)

Test 1: the Loss object directly.
Test 2: the real Connect4 model, compiled exactly like Train.py, via
        evaluate() and train_on_batch() with the Dataloader's (w, w) structure.
Test 3: gradient-level proof: a weight-0 sample with a huge error. If weights
        are per-sample its gradient must vanish; if broadcast it does not.
"""
import sys, os
sys.path.insert(0, r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

from Net.Alpha_Loss import Policy_Loss, Value_Loss

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- Test 1
print("=" * 70)
print("Test 1: Value_Loss object called with sample_weight")
y_true = np.array([[1.0], [-1.0], [0.5], [0.0]], dtype=np.float32)
y_pred = np.array([[0.0], [-1.0], [0.0], [2.0]], dtype=np.float32)
w = np.array([4.0, 2.0, 1.0, 0.0], dtype=np.float32)  # mean 1.75
se = ((y_true - y_pred) ** 2).ravel()

H_per_sample = float(np.sum(w * se) / len(se))
H_broadcast = float(np.mean(se) * np.mean(w))

vl = Value_Loss(dtype="float32")
got = float(vl(y_true, y_pred, sample_weight=w))
print(f"  per-sample SEs: {se.tolist()}, weights: {w.tolist()}")
print(f"  H_per_sample = {H_per_sample:.6f}")
print(f"  H_broadcast  = {H_broadcast:.6f}")
print(f"  Value_Loss   = {got:.6f}")
which = ("PER-SAMPLE" if abs(got - H_per_sample) < 1e-6 else
         "BROADCAST (weights are a no-op per-sample)" if abs(got - H_broadcast) < 1e-6
         else "NEITHER?!")
print(f"  -> {which}")

# reference: a correct per-sample loss for comparison
class Correct_Value_Loss(tf.keras.Loss):
    def call(self, y_true, y_pred):
        return tf.square(y_true - y_pred)  # per-sample, let keras reduce
ref = float(Correct_Value_Loss(dtype="float32")(y_true, y_pred, sample_weight=w))
print(f"  reference per-sample-returning loss gives: {ref:.6f}")

# ---------------------------------------------------------------- Test 2
print("=" * 70)
print("Test 2: real Connect4 model + Train.py compile, evaluate with (w, w)")
from Connect4.game import Connect4
from Connect4.Build_Model import build_model
from Connect4.config import build_config, train_config

game = Connect4()
model = build_model(game.get_input_state().shape, game.policy_shape,
                    (build_config, train_config, None))
model.compile(optimizer=tf.keras.optimizers.SGD(0.0),
              loss={"policy": Policy_Loss(dtype="float32"),
                    "value": Value_Loss(dtype="float32")})

N = 8
states = rng.integers(-1, 2, size=(N, 6, 7, 4)).astype(np.int8)
pols = rng.random((N, 7)).astype(np.float32)
pols /= pols.sum(1, keepdims=True)
vals = rng.uniform(-1, 1, size=(N, 1)).astype(np.float32)
w = np.array([8, 4, 2, 1, 1, 0, 0, 0], dtype=np.float32)  # mean 2.0

p_hat, v_hat = model.predict(states.astype(np.float32), verbose=0)
se = ((vals - v_hat) ** 2).ravel()
ce = -(pols * np.log(np.clip(p_hat, 1e-7, 1 - 1e-7))).sum(1)

H_v_per_sample = float(np.sum(w * se) / N)
H_v_broadcast = float(np.mean(se) * np.mean(w))
H_p_per_sample = float(np.sum(w * ce) / N)
H_p_broadcast = float(np.mean(ce) * np.mean(w))

out = model.evaluate(states, (pols, vals), sample_weight=(w, w),
                     batch_size=N, verbose=0, return_dict=True)
print(f"  keras value loss  = {out['value_loss']:.6f}")
print(f"  H_v_per_sample    = {H_v_per_sample:.6f}")
print(f"  H_v_broadcast     = {H_v_broadcast:.6f}")
print(f"  keras policy loss = {out['policy_loss']:.6f}")
print(f"  H_p_per_sample    = {H_p_per_sample:.6f}")
print(f"  H_p_broadcast     = {H_p_broadcast:.6f}")

def judge(got, ps, bc):
    if abs(got - ps) < 5e-4: return "PER-SAMPLE"
    if abs(got - bc) < 5e-4: return "BROADCAST"
    return "NEITHER"
print(f"  -> value head: {judge(out['value_loss'], H_v_per_sample, H_v_broadcast)}, "
      f"policy head: {judge(out['policy_loss'], H_p_per_sample, H_p_broadcast)}")

# ---------------------------------------------------------------- Test 3
print("=" * 70)
print("Test 3: does a weight-0 sample still push gradients?")
# two samples: sample A weight 0 with a huge value error, sample B weight 1 with
# zero error. Per-sample weighting => total gradient ~ 0. Broadcast => big grad.
states2 = np.stack([states[0], states[1]]).astype(np.float32)
p_hat2, v_hat2 = model.predict(states2, verbose=0)
pols2 = p_hat2.copy()                       # zero policy error
vals2 = np.array([[1.0], [0.0]], dtype=np.float32)
vals2[1] = v_hat2[1]                        # sample B: zero value error
w2 = np.array([0.0, 1.0], dtype=np.float32)  # sample A silenced

x = tf.constant(states2)
with tf.GradientTape() as tape:
    p_out, v_out = model(x, training=False)
    lv = Value_Loss(dtype="float32")(tf.constant(vals2), v_out,
                                     sample_weight=tf.constant(w2))
g = tape.gradient(lv, model.trainable_variables)
gnorm = float(tf.linalg.global_norm([t for t in g if t is not None]))
print(f"  value error of silenced sample: {float(vals2[0] - v_hat2[0]):+.4f}")
print(f"  grad norm with weight-0 on the erroring sample: {gnorm:.6f}")
print(f"  -> {'weights DO silence samples (per-sample)' if gnorm < 1e-6 else 'weight-0 sample STILL trains the net (broadcast)'}")
