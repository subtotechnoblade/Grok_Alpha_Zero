"""Verify the four convergence fixes end to end.

1. tau off-by-one: truth table for both players, old vs new formula
2. dedup_average: synthetic duplicates -> counts, means, weight normalization
3. real data: weights on the actual gen window, empty-board weight restored
4. keras plumbing: model.fit/evaluate accept the (x, y, sample_weight) batches
   with the real custom losses, and the weights measurably change the loss
"""
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

print("=" * 70)
print("1. tau explore plies (config says 3 per player)")
print("=" * 70)
n = 3
print(f"{'ply':>4} {'player':>7} {'old formula':>12} {'new formula':>12}")
for ply in range(8):
    if ply % 2 == 0:
        old = new = ply // 2 < n
        who = "-1"
    else:
        old = (ply + 1) // 2 < n
        new = (ply - 1) // 2 < n
        who = "+1"
    print(f"{ply:>4} {who:>7} {str(old):>12} {str(new):>12}")
p1_old = sum((p + 1) // 2 < n for p in (1, 3, 5, 7))
p1_new = sum((p - 1) // 2 < n for p in (1, 3, 5, 7))
assert p1_new == 3 and p1_old == 2
print(f"player +1 explore plies: old {p1_old} (BUG), new {p1_new} (matches config)\n")

print("=" * 70)
print("2. dedup_average returns counts (synthetic)")
print("=" * 70)
from Dataloader import dedup_average
A = np.zeros((6, 7, 4), dtype=np.int8)
B = np.ones((6, 7, 4), dtype=np.int8)
states = [A, A, A, B]                       # A x3, B x1
pols = [np.array([1, 0], np.float32), np.array([0, 1], np.float32),
        np.array([1, 0], np.float32), np.array([0.5, 0.5], np.float32)]
vals = [np.array([1.0], np.float32), np.array([0.0], np.float32),
        np.array([-1.0], np.float32), np.array([0.5], np.float32)]
s, p, v, c = dedup_average(states, pols, vals)
order = np.argsort(c)[::-1]                 # A (count 3) first
assert len(s) == 2 and c[order[0]] == 3.0 and c[order[1]] == 1.0
assert np.allclose(p[order[0]], [2 / 3, 1 / 3]) and np.isclose(v[order[0]][0], 0.0)
w = c * (len(c) / c.sum())
print(f"counts {c[order]} -> normalized weights {w[order].round(3)} (mean {w.mean():.3f})")
print("means correct, counts correct\n")

print("=" * 70)
print("3. real window: split with dedup + weights")
print("=" * 70)
from Connect4.game import Connect4
from Connect4.config import train_config
from Dataloader import Create_Train_Test_Split, Dataloader

((ts, tp, tv, tw), (vs, vp, vv, vw)) = Create_Train_Test_Split(
    "Connect4/Grok_Zero_Train",
    num_previous_generations=train_config["num_previous_generations"],
    train_percent=train_config["train_percent"],
    train_decay=train_config["train_decay"],
    test_percent=train_config["test_percent"],
    test_decay=train_config["test_decay"],
    dedup_average=True).split()

empty = Connect4().get_input_state()
idx = next(i for i, st in enumerate(ts) if np.array_equal(st, empty))
share = tw[idx] / tw.sum()
print(f"train weights: n={len(tw)}, mean={tw.mean():.4f} (must be 1.0), max={tw.max():.0f}")
print(f"empty board: weight {tw[idx]:.0f} -> {share:.2%} of total training mass "
      f"(was 0.0027% unweighted, ~2.8% pre-dedup)")
assert abs(tw.mean() - 1.0) < 1e-4 and tw[idx] > 100

print()
print("=" * 70)
print("4. keras accepts weighted batches with the real losses")
print("=" * 70)
import tensorflow as tf
from Connect4.Build_Model import build_model
from Connect4.config import build_config, optimizer_config
from Net.Alpha_Loss import Policy_Loss, Value_Loss

k = 2048                                    # small slice, CPU friendly
dl = Dataloader(ts[:k], tp[:k], tv[:k], 256, weights=tw[:k])
x, y, sw = dl[0]
print(f"batch structure: x{x.shape}, y=({y[0].shape}, {y[1].shape}), "
      f"sw=({sw[0].shape}, {sw[1].shape})")

g = Connect4()
model = build_model(g.get_input_state().shape, g.policy_shape,
                    (build_config, train_config, optimizer_config))
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss={"policy": Policy_Loss(dtype="float32"),
                    "value": Value_Loss(dtype="float32")})

hist = model.fit(dl, epochs=1, verbose=0)
loss_w = model.evaluate(dl, verbose=0, return_dict=True)["loss"]
dl_unw = Dataloader(ts[:k], tp[:k], tv[:k], 256)   # same data, uniform weights
loss_u = model.evaluate(dl_unw, verbose=0, return_dict=True)["loss"]
print(f"fit ran, final training loss {hist.history['loss'][0]:.4f} (finite: "
      f"{np.isfinite(hist.history['loss'][0])})")
print(f"evaluate weighted {loss_w:.4f} vs uniform {loss_u:.4f} -> weights are "
      f"{'APPLIED' if abs(loss_w - loss_u) > 1e-4 else 'IGNORED (BUG)'}")
assert np.isfinite(hist.history["loss"][0]) and abs(loss_w - loss_u) > 1e-4
print("\nALL CHECKS PASSED")
