"""Where in the game does the value head lose, and does it shrink toward zero?

For model g evaluated on generation g's data (held out for it by construction),
per ply band:
  MSE            weighted by occurrence, the training objective
  mean/std of the dedup-averaged TARGET vs the PREDICTION
  corr           correlation between them
  beta           slope of target on prediction. beta > 1 means the net's output
                 is SHRUNK: it has the sign right but not the magnitude.
  floor          irreducible label noise after dedup averaging

  usage: shrinkage.py <run_root> <gen> [<gen> ...]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
os.chdir(ROOT)

import numpy as np
from gtlib import (iter_states, dedup_average, run_model, session,
                   ply_of, band_of, BAND_NAMES)

run_root = sys.argv[1]
# "12" = model 12 on data 12 (held out).  "12:11" = model 12 on data 11 (INSIDE
# its own training window -> a fit number). Comparing the two separates a
# fitting failure from a generalization failure.
pairs = []
for a in sys.argv[2:]:
    if ":" in a:
        m, d = a.split(":")
        pairs.append((int(m), int(d)))
    else:
        pairs.append((int(a), int(a)))

print(f"run: {run_root}")
print(f"{'mdl':>4} {'data':>5} {'band':>7} {'n':>7} {'MSE':>8} {'MSE(V=0)':>9} "
      f"{'vs V=0':>7} {'tgt std':>8} {'pred std':>9} {'corr':>6} {'beta':>6} {'floor':>7}")

for g, dg in pairs:
    fp = f"{run_root}/{dg}/Self_Play_Data.h5"
    onnx = f"{run_root}/{g}/model.onnx"
    if not (os.path.exists(fp) and os.path.exists(onnx)):
        continue
    X, Y = [], []
    for b, _, v in iter_states(fp, stride=2):
        X.append(b)
        Y.append(np.asarray(v).reshape(-1))
    X, Y = np.concatenate(X), np.concatenate(Y)

    flat = X.reshape(len(X), -1)
    uniq, inv, counts = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros(len(uniq)); sqs = np.zeros(len(uniq))
    np.add.at(sums, inv, Y); np.add.at(sqs, inv, Y ** 2)
    T = sums / counts
    var_within = np.maximum(sqs / counts - T ** 2, 0.0)
    Xu = uniq.reshape(-1, *X.shape[1:])

    sess = session(onnx)
    _, V = run_model(sess, Xu.astype(np.float32))

    bands = np.array([band_of(ply_of(x[..., -1])) for x in Xu])
    w = counts / counts.mean()

    for i, name in list(enumerate(BAND_NAMES)) + [(-1, "ALL")]:
        m = np.ones(len(Xu), bool) if i == -1 else (bands == i)
        if m.sum() < 10:
            continue
        t, p, ww = T[m], V[m], w[m]
        mse = float(np.sum(ww * (p - t) ** 2) / np.sum(ww))
        mse0 = float(np.sum(ww * t ** 2) / np.sum(ww))       # constant-zero predictor
        floor = float(np.sum(ww * var_within[m] / counts[m]) / np.sum(ww))
        sp, st = p.std(), t.std()
        corr = float(np.corrcoef(p, t)[0, 1]) if sp > 1e-6 and st > 1e-6 else float("nan")
        beta = float(np.cov(t, p)[0, 1] / max(sp ** 2, 1e-12)) if sp > 1e-6 else float("nan")
        print(f"{g:>4} {dg:>5} {name:>7} {int(m.sum()):>7} {mse:>8.4f} {mse0:>9.4f} "
              f"{1 - mse / max(mse0, 1e-9):>7.1%} {st:>8.3f} {sp:>9.3f} {corr:>6.3f} "
              f"{beta:>6.2f} {floor:>7.4f}")
    print()
