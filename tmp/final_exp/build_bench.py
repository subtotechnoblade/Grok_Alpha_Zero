"""Freeze one shared ground-truth benchmark used by EVERY arm.

Each arm generates its own positions, so an "own data" metric compares models on
different position sets. This builds one fixed set, drawn from the archived
run2/run3 self play (real Connect4 positions, produced by neither arm, present
in no arm's training data because every arm starts from generation 0).

  win-in-1  : mover can complete 4-in-a-row -> true value = +1
  must-block: mover has no win, opponent has exactly one winning column
              -> that column is forced (policy control)

Deduplicated on the full state tensor. Written to bench.npz.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tmp", "final_exp"))
os.chdir(ROOT)

import numpy as np
from gtlib import collect_ground_truth, ply_of, band_of, BAND_NAMES

SOURCES = []
for root, gens in (("tmp/run2_archive_gen0-19_oldvaluehead", range(4, 20)),
                   ("tmp/run3_archive_gen0-15_bottleneck_preweightfix", range(4, 16))):
    for g in gens:
        p = f"{root}/{g}/Self_Play_Data.h5"
        if os.path.exists(p):
            SOURCES.append(p)

CAP = 6000

winX, winCols, winMask, winPly = [], [], [], []
blkX, blkCol, blkMask, blkPly = [], [], [], []

for p in SOURCES:
    w, b = collect_ground_truth(p, stride=3)      # stride: spread over games
    winX += w["X"]; winCols += w["cols"]; winMask += w["mask"]; winPly += w["ply"]
    blkX += b["X"]; blkCol += b["col"]; blkMask += b["mask"]; blkPly += b["ply"]
    print(f"{p}: +{len(w['X'])} win, +{len(b['X'])} block "
          f"(running {len(winX)}/{len(blkX)})", flush=True)

def dedup(X, extra):
    X = np.asarray(X, dtype=np.int8)
    _, idx = np.unique(X.reshape(len(X), -1), axis=0, return_index=True)
    idx = np.sort(idx)
    return X[idx], [[e[i] for i in idx] for e in extra]

winX, (winCols, winMask, winPly) = dedup(winX, (winCols, winMask, winPly))
blkX, (blkCol, blkMask, blkPly) = dedup(blkX, (blkCol, blkMask, blkPly))
print(f"after dedup: {len(winX)} win-in-1, {len(blkX)} must-block")

rng = np.random.default_rng(0)
if len(winX) > CAP:
    keep = np.sort(rng.choice(len(winX), CAP, replace=False))
    winX = winX[keep]; winCols = [winCols[i] for i in keep]
    winMask = [winMask[i] for i in keep]; winPly = [winPly[i] for i in keep]
if len(blkX) > CAP:
    keep = np.sort(rng.choice(len(blkX), CAP, replace=False))
    blkX = blkX[keep]; blkCol = [blkCol[i] for i in keep]
    blkMask = [blkMask[i] for i in keep]; blkPly = [blkPly[i] for i in keep]

# pad winning-column lists into a fixed-width mask
winColMask = np.zeros((len(winX), 7), dtype=bool)
for i, cols in enumerate(winCols):
    winColMask[i, cols] = True

np.savez_compressed(
    "tmp/final_exp/bench.npz",
    winX=winX, winColMask=winColMask,
    winMask=np.asarray(winMask, dtype=bool), winPly=np.asarray(winPly),
    blkX=blkX, blkCol=np.asarray(blkCol), blkMask=np.asarray(blkMask, dtype=bool),
    blkPly=np.asarray(blkPly))

print(f"\nFROZEN BENCHMARK  win-in-1 {len(winX)}   must-block {len(blkX)}")
for name, plys in (("win-in-1", winPly), ("must-block", blkPly)):
    counts = np.zeros(len(BAND_NAMES), dtype=int)
    for p in plys:
        counts[band_of(p)] += 1
    print(f"  {name:<11}" + "  ".join(f"{b}:{c}" for b, c in zip(BAND_NAMES, counts)))
print("  mean legal moves win-in-1:",
      round(float(np.asarray(winMask).sum(1).mean()), 2),
      " -> random-policy top1 baseline ~",
      round(float(np.mean([m.sum() and (np.asarray(winColMask[i]) & winMask[i]).sum() / winMask[i].sum()
                           for i, m in enumerate(winMask)])), 3))
print("  random-policy must-block top1 baseline ~",
      round(float(np.mean(1.0 / np.asarray(blkMask).sum(1))), 3))
