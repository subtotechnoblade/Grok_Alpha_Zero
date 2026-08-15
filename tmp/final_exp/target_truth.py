"""Is the value TARGET right on positions with a known answer?

Separates two very different failures:
  (a) the stored target already says +1 on a win-in-1 and the net does not fit it
      -> a fitting / coverage problem
  (b) the stored target does NOT say +1
      -> the labels are wrong and no amount of training or data can help

Also decomposes the value-label noise, and computes the two different MSE floors
that a perfect predictor faces:
  raw    : E[Var(z | state)]                        -- floor on RAW samples
  dedup  : the floor after Dataloader.dedup_average replaces each repeated state
           by the MEAN of its labels, which is what val_value_loss is measured
           against. Averaging n labels cuts their variance by n, so this floor is
           far LOWER than the raw one -- which is why a val loss can legitimately
           land below a floor estimated the raw way.

  usage: target_truth.py <run_root> <gen> [<gen> ...]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
os.chdir(ROOT)

import numpy as np
from gtlib import (collect_ground_truth, iter_states, win_cols, mover_of,
                   ply_of, band_of, BAND_NAMES, legal_mask)

run_root = sys.argv[1]
gens = [int(x) for x in sys.argv[2:]]

print(f"run: {run_root}")
print()
print("A. STORED TARGETS on positions with a known answer (mover's perspective)")
print(f"{'gen':>4} {'n_win':>6} {'target mean':>12} {'>0':>7} {'>0.9':>7} | "
      f"{'n_blk':>6} {'policy mass on forced col':>26} {'argmax ok':>10}")
for g in gens:
    fp = f"{run_root}/{g}/Self_Play_Data.h5"
    if not os.path.exists(fp):
        continue
    wv, bm, bt = [], [], []
    for boards, pol, val in iter_states(fp, stride=2):
        val = np.asarray(val).reshape(-1)
        for i, st in enumerate(boards):
            board = st[..., -1]
            mover = mover_of(st)
            mine = win_cols(board, mover)
            if mine:
                wv.append(val[i])
                continue
            theirs = win_cols(board, -mover)
            if len(theirs) == 1:
                p = np.asarray(pol[i], dtype=np.float64)
                lm = legal_mask(board)
                p = p * lm
                s = p.sum()
                p = p / s if s > 0 else p
                bm.append(p[theirs[0]])
                bt.append(int(np.argmax(p)) == theirs[0])
    wv = np.asarray(wv)
    if not len(wv):
        continue
    print(f"{g:>4} {len(wv):>6} {wv.mean():>+12.3f} {np.mean(wv > 0):>7.1%} "
          f"{np.mean(wv > 0.9):>7.1%} | {len(bm):>6} {np.mean(bm):>26.1%} "
          f"{np.mean(bt):>10.1%}")

print()
print("B. VALUE-LABEL NOISE AND THE TWO MSE FLOORS")
print(f"{'gen':>4} {'n_raw':>8} {'n_uniq':>8} {'dup%':>6} {'meanGrp':>8} "
      f"{'Var(z)':>8} {'floor_raw':>10} {'floor_dedup':>12} {'Var(target_dedup)':>18}")
for g in gens:
    fp = f"{run_root}/{g}/Self_Play_Data.h5"
    if not os.path.exists(fp):
        continue
    X, Y = [], []
    for b, _, v in iter_states(fp, stride=2):
        X.append(b)
        Y.append(np.asarray(v).reshape(-1))
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    flat = X.reshape(len(X), -1)
    uniq, inv, counts = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros(len(uniq))
    sqs = np.zeros(len(uniq))
    np.add.at(sums, inv, Y)
    np.add.at(sqs, inv, Y ** 2)
    means = sums / counts
    var_within = np.maximum(sqs / counts - means ** 2, 0.0)      # per unique state

    # raw floor: a perfect predictor still eats the within-state spread of labels
    floor_raw = float(np.sum(var_within * counts) / counts.sum())
    # dedup floor: the target is the group MEAN, whose variance is var/n;
    # training weights each unique state by its count, normalized to mean 1
    w = counts / counts.mean()
    floor_dedup = float(np.sum(w * var_within / counts) / w.sum())
    print(f"{g:>4} {len(X):>8} {len(uniq):>8} "
          f"{1 - len(uniq) / len(X):>6.1%} {counts.mean():>8.2f} "
          f"{float(np.var(Y)):>8.3f} {floor_raw:>10.4f} {floor_dedup:>12.4f} "
          f"{float(np.var(means)):>18.3f}")

print()
print("C. WITHIN-STATE LABEL SPREAD BY PLY BAND (std of z for one identical state)")
print(f"{'gen':>4} " + " ".join(f"{b:>10}" for b in BAND_NAMES))
for g in gens:
    fp = f"{run_root}/{g}/Self_Play_Data.h5"
    if not os.path.exists(fp):
        continue
    X, Y = [], []
    for b, _, v in iter_states(fp, stride=2):
        X.append(b)
        Y.append(np.asarray(v).reshape(-1))
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    flat = X.reshape(len(X), -1)
    uniq, inv, counts = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros(len(uniq)); sqs = np.zeros(len(uniq))
    np.add.at(sums, inv, Y); np.add.at(sqs, inv, Y ** 2)
    means = sums / counts
    var_within = np.maximum(sqs / counts - means ** 2, 0.0)
    bands = np.array([band_of(ply_of(u.reshape(6, 7, 4)[..., -1])) for u in uniq])
    out = []
    for i in range(len(BAND_NAMES)):
        m = (bands == i) & (counts > 1)
        out.append(f"{np.sqrt(var_within[m].mean()):>10.3f}" if m.any() else f"{'-':>10}")
    print(f"{g:>4} " + " ".join(out))
