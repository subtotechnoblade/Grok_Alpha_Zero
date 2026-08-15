"""ITEM 1 + 2: loss semantics, weight normalization, gradient scaling, metric safety.

Builds the REAL Connect4 model and compiles it through the REAL Train.train code
path (fit/load_weights/os.remove monkeypatched to no-ops so nothing trains and
nothing is written).
"""
import os, sys, math
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
REPO = r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero"
sys.path.insert(0, REPO)

import numpy as np
import tensorflow as tf
import keras

np.random.seed(0)
tf.random.set_seed(0)

from Connect4.game import Connect4
from Connect4.config import build_config, train_config, optimizer_config, lr_schedule
from Connect4.Build_Model import build_model
import Train as TrainMod

game = Connect4()
IN_SHAPE = game.get_input_state().shape
POLICY_SHAPE = game.policy_shape
print("input_shape", IN_SHAPE, "policy_shape", POLICY_SHAPE,
      "state dtype", game.get_input_state().dtype)

configs = (build_config, train_config, optimizer_config)


def compile_like_train(model):
    """Run the exact Train.train compile block."""
    real_fit = tf.keras.Model.fit
    real_load = tf.keras.Model.load_weights
    real_remove = os.remove
    tf.keras.Model.fit = lambda self, *a, **k: None
    tf.keras.Model.load_weights = lambda self, *a, **k: None
    os.remove = lambda p: None
    try:
        TrainMod.train(None, None, model, lr_schedule(0), configs,
                       run_dir=os.path.dirname(os.path.abspath(__file__)))
    finally:
        tf.keras.Model.fit = real_fit
        tf.keras.Model.load_weights = real_load
        os.remove = real_remove
    return model


model = build_model(IN_SHAPE, POLICY_SHAPE, configs)
compile_like_train(model)
print("compiled. optimizer:", type(model.optimizer).__name__,
      "loss:", type(model._compile_loss).__name__)
cl = model._compile_loss
print("compile_loss.reduction =", cl.reduction)
for fl in cl._flat_losses if cl._flat_losses else []:
    print("   flat loss:", fl.path, type(fl.loss).__name__, "w=", fl.loss_weights)

# ---------------------------------------------------------------- data batch
N = 8
X = np.random.randint(-1, 2, size=(N, *IN_SHAPE)).astype(np.int8)
Praw = np.random.random((N, POLICY_SHAPE[0])).astype(np.float64)
P = (Praw / Praw.sum(1, keepdims=True)).astype(np.float32)
V = np.random.uniform(-1, 1, size=(N, 1)).astype(np.float32)
# deliberately NOT mean-1 so /N and /sum(w) differ
W = np.array([0.0, 1.0, 2.0, 4.0, 0.5, 3.0, 1.0, 0.25], dtype=np.float32)
print("\nweights", W, "sum", W.sum(), "N", N, " (mean = %.4f)" % W.mean())

y_pred = model(X, training=False)
pp = np.asarray(y_pred[0], dtype=np.float64)
vp = np.asarray(y_pred[1], dtype=np.float64)

# hand-computed per-sample losses, replicating keras' normalize+clip in CCE
eps = keras.backend.epsilon()
pn = pp / pp.sum(1, keepdims=True)
pn = np.clip(pn, eps, 1.0 - eps)
ce = -(P.astype(np.float64) * np.log(pn)).sum(1)          # (N,)
mse = ((V.astype(np.float64) - vp) ** 2).mean(1)          # (N,)
ce_keras = np.asarray(keras.losses.categorical_crossentropy(P, y_pred[0]), np.float64)
print("hand CE vs keras functional CE max abs diff:",
      np.abs(ce - ce_keras).max())

cand = {
    "sum(w*l)/N        ": (W * ce).sum() / N + (W * mse).sum() / N,
    "sum(w*l)/sum(w)   ": (W * ce).sum() / W.sum() + (W * mse).sum() / W.sum(),
    "sum(w*l)          ": (W * ce).sum() + (W * mse).sum(),
    "unweighted sum/N  ": ce.mean() + mse.mean(),
}

sw = (tf.constant(W), tf.constant(W))
loss_tf = model.compute_loss(x=X, y=(P, V), y_pred=y_pred, sample_weight=sw)
loss_tf = float(np.asarray(loss_tf))
print("\nmodel.compute_loss  = %.10f" % loss_tf)
for k, v in cand.items():
    print("   %s = %.10f   %s" % (k, v, "  <== MATCH" if abs(v - loss_tf) < 1e-5 else ""))

# per-head split
pl = float(np.asarray(cl._flat_losses[0].loss(P, y_pred[0], tf.constant(W))))
vl = float(np.asarray(cl._flat_losses[1].loss(V, y_pred[1], tf.constant(W))))
print("policy head loss %.10f  (hand sum(w*ce)/N %.10f)" % (pl, (W * ce).sum() / N))
print("value  head loss %.10f  (hand sum(w*mse)/N %.10f)" % (vl, (W * mse).sum() / N))
print("policy+value = %.10f   total = %.10f   equal: %s" %
      (pl + vl, loss_tf, abs(pl + vl - loss_tf) < 1e-6))

# ------------------------------------------------- does the loss return per-sample?
from Net.Alpha_Loss import Policy_Loss, Value_Loss
plf, vlf = Policy_Loss(dtype="float32"), Value_Loss(dtype="float32")
print("\nPolicy_Loss.call(...) shape:", plf.call(tf.constant(P), y_pred[0]).shape,
      " reduction:", plf.reduction)
print("Value_Loss.call(...)  shape:", vlf.call(tf.constant(V), y_pred[1]).shape,
      " reduction:", vlf.reduction)

# ------------------------------------------------------------- gradient tests
def grads_for(x, p, v, w, training=False):
    with tf.GradientTape() as tape:
        yp = model(x, training=training)
        L = model.compute_loss(x=x, y=(p, v), y_pred=yp,
                               sample_weight=(tf.constant(w), tf.constant(w)))
    g = tape.gradient(L, model.trainable_variables)
    return [np.zeros_like(vv.numpy()) if gi is None else gi.numpy()
            for gi, vv in zip(g, model.trainable_variables)]


def gnorm(gs):
    return math.sqrt(sum(float((g ** 2).sum()) for g in gs))


x1, p1, v1 = X[:1], P[:1], V[:1]
g_w1 = grads_for(x1, p1, v1, np.array([1.0], np.float32))
g_w4 = grads_for(x1, p1, v1, np.array([4.0], np.float32))
print("\n--- single sample, weight 1 vs weight 4 (N=1 so loss = w*l) ---")
print("||g(w=1)|| = %.8e" % gnorm(g_w1))
print("||g(w=4)|| = %.8e" % gnorm(g_w4))
print("ratio = %.8f   (expected 4.0)" % (gnorm(g_w4) / gnorm(g_w1)))
maxrel = max(float(np.abs(b - 4 * a).max() / (np.abs(4 * a).max() + 1e-30))
             for a, b in zip(g_w1, g_w4) if np.abs(a).max() > 0)
print("max elementwise rel err of g(w=4) vs 4*g(w=1): %.3e" % maxrel)

# weight 0 -> zero contribution
x2 = np.concatenate([X[:1], X[1:2]])
p2 = np.concatenate([P[:1], P[1:2]])
v2 = np.concatenate([V[:1], V[1:2]])
g_pair = grads_for(x2, p2, v2, np.array([1.0, 0.0], np.float32))
g_solo_half = [0.5 * g for g in g_w1]   # N=2 divisor
print("\n--- batch of 2 with weights [1, 0] vs single sample /2 ---")
print("||g(pair, w=[1,0])||   = %.8e" % gnorm(g_pair))
print("||0.5 * g(solo, w=1)|| = %.8e" % gnorm(g_solo_half))
diff = max(float(np.abs(a - b).max()) for a, b in zip(g_pair, g_solo_half))
scale = max(float(np.abs(b).max()) for b in g_solo_half)
print("max abs diff %.3e  (max grad magnitude %.3e) -> rel %.3e" %
      (diff, scale, diff / (scale + 1e-30)))

# perturb the zero-weight sample's targets: gradients must not move at all
p2b = p2.copy(); p2b[1] = np.eye(7, dtype=np.float32)[0]
v2b = v2.copy(); v2b[1] = -v2[1]
g_pair_b = grads_for(x2, p2b, v2b, np.array([1.0, 0.0], np.float32))
d2 = max(float(np.abs(a - b).max()) for a, b in zip(g_pair, g_pair_b))
print("changing the w=0 sample's targets moves gradients by max %.3e" % d2)

# ---------------------------------------- item 2: weighted_metrics side effects
model2 = build_model(IN_SHAPE, POLICY_SHAPE, configs)
model2.set_weights(model.get_weights())
opt2 = TrainMod.Orthograd(TrainMod.Muon(**{**optimizer_config["kwargs"],
                                           "learning_rate": lr_schedule(0)}))
from Net.Alpha_Loss import Policy_Loss as PL2, Value_Loss as VL2
model2.compile(optimizer=opt2,
               loss={"policy": PL2(dtype="float32"), "value": VL2(dtype="float32")})
with tf.GradientTape() as tape:
    yp = model2(X, training=False)
    L2 = model2.compute_loss(x=X, y=(P, V), y_pred=yp,
                             sample_weight=(tf.constant(W), tf.constant(W)))
g2 = tape.gradient(L2, model2.trainable_variables)
g2 = [np.zeros_like(v.numpy()) if gi is None else gi.numpy()
      for gi, v in zip(g2, model2.trainable_variables)]
gfull = grads_for(X, P, V, W)
print("\n--- weighted_metrics present vs absent ---")
print("loss with metrics    = %.10f" % loss_tf)
print("loss without metrics = %.10f" % float(np.asarray(L2)))
dm = max(float(np.abs(a - b).max()) for a, b in zip(gfull, g2))
print("max abs gradient difference: %.3e" % dm)

# -------------------------------------------------- val_loss composition
sys.path.insert(0, REPO)
from Dataloader import Dataloader
dl = Dataloader(list(X), list(P), list(V), 4, weights=W)
res = model.evaluate(dl, return_dict=True, verbose=0)
print("\n--- model.evaluate(dataloader) keys ---")
for k, v in res.items():
    print("   %-14s %.10f" % (k, v))
