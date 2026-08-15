"""Empirically verify the claim: a bias followed by BatchNorm never moves off zero.

Reads the gen-18 Keras weights (saved BEFORE the use_bias=False change, so every
bias is still present) and reports max|value| for every 1-D tensor, grouped by
whether it is a bias or a BN parameter.
"""
import sys

import numpy as np
import h5py as h5

PATH = r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero\Connect4\Grok_Zero_Train\18\model.weights.h5"

rows = []


def walk(name, obj):
    if isinstance(obj, h5.Dataset):
        rows.append((name, obj.shape, np.asarray(obj)))


with h5.File(PATH, "r") as f:
    f.visititems(walk)

print(f"{len(rows)} weight tensors in {PATH.split(chr(92))[-1]}\n")

# Keras 3 stores vars positionally; show the full path so layer identity is visible.
print(f"{'path':<74} {'shape':>14} {'max|w|':>11} {'nonzero':>8}")
for name, shape, arr in rows:
    a = np.asarray(arr, dtype=np.float64)
    print(f"{name[:74]:<74} {str(shape):>14} {np.abs(a).max():>11.3e} "
          f"{np.mean(a != 0):>7.1%}")
