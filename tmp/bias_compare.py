"""Live output-head biases vs the dead BN-shadowed ones.

Magnitude alone can't separate them -- a random walk grows too. The tell is
STRUCTURE:

  * policy output bias (7 values, feeds softmax): training augments every game
    with its mirror, so a genuinely learned column bias must come out roughly
    left-right symmetric, b[i] ~ b[6-i]. Noise has no reason to be.
    Note softmax is shift invariant, so only b - mean(b) does anything: 6 real
    degrees of freedom, not 7.
  * value output bias (1 value, feeds tanh): fully live, shifts every eval.
  * dead biases: should look like zero-mean white noise growing as sqrt(steps).

Tracked across generations: a learned parameter trends/settles, a random walk wanders.
"""
import os
import sys
import glob

import numpy as np
import h5py as h5

ROOT = r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero"

LIVE = {"policy out (dense_4)": "layers/dense_4/vars/1",
        "value  out (dense_5)": "layers/dense_5/vars/1"}
DEAD = {"eyes conv (conv2d)": "layers/conv2d/vars/1",
        "resblock0 conv1": "layers/res_net__block/conv1/vars/1",
        "policy Dense128 (dense)": "layers/dense/vars/1",
        "value  Dense128 (dense_1)": "layers/dense_1/vars/1"}

gens = sorted(int(os.path.basename(os.path.dirname(p)))
              for p in glob.glob(f"{ROOT}/Connect4/Grok_Zero_Train/*/model.weights.h5"))


def load(gen, key):
    with h5.File(f"{ROOT}/Connect4/Grok_Zero_Train/{gen}/model.weights.h5", "r") as f:
        return np.asarray(f[key], dtype=np.float64) if key in f else None


latest = gens[-1]
print(f"=== gen {latest}: the LIVE output-head biases, actual values ===\n")

pb = load(latest, LIVE["policy out (dense_4)"])
print("policy output bias, per column (pre-softmax logit offset):")
print("   col:  " + "  ".join(f"{c:>7d}" for c in range(7)))
print("   b  :  " + "  ".join(f"{x:>+7.4f}" for x in pb))
print("   b - mean(b) (the part softmax actually sees):")
print("         " + "  ".join(f"{x:>+7.4f}" for x in pb - pb.mean()))
mirror = pb[::-1]
print(f"\n   mirror symmetry check  b[i] vs b[6-i]:")
print(f"     correlation with own mirror : {np.corrcoef(pb, mirror)[0, 1]:+.4f}")
print(f"     max |b[i] - b[6-i]|         : {np.abs(pb - mirror).max():.3e}")
print(f"     relative asymmetry          : "
      f"{np.abs(pb - mirror).max() / (np.abs(pb).max() + 1e-12):.1%}")

vb = load(latest, LIVE["value  out (dense_5)"])
print(f"\nvalue output bias (scalar, pre-tanh): {vb[0]:+.5f}  -> tanh = {np.tanh(vb[0]):+.5f}")
print("   i.e. the net's board-independent lean, in value units")

print()
print("=== magnitude comparison at gen", latest, "===")
print(f"{'tensor':<28} {'n':>5} {'max|b|':>11} {'rms':>11} {'mean':>11} {'status':>8}")
for label, key in list(LIVE.items()) + list(DEAD.items()):
    a = load(latest, key)
    if a is None:
        continue
    tag = "LIVE" if label in LIVE else "dead"
    print(f"{label:<28} {a.size:>5} {np.abs(a).max():>11.3e} "
          f"{np.sqrt((a ** 2).mean()):>11.3e} {a.mean():>+11.3e} {tag:>8}")

print()
print("=== trajectory across generations: rms magnitude ===")
print("   a learned param settles; a random walk keeps growing ~sqrt(t)")
hdr = f"{'gen':>4} "
keys = list(LIVE.items()) + list(DEAD.items())
for label, _ in keys:
    hdr += f"{label.split('(')[0].strip()[:13]:>15}"
print(hdr)
for g in gens:
    line = f"{g:>4} "
    for _, key in keys:
        a = load(g, key)
        line += f"{np.sqrt((a ** 2).mean()):>15.3e}" if a is not None else f"{'-':>15}"
    print(line)

print()
print("=== policy-bias mirror asymmetry over time (should FALL if learned) ===")
print(f"{'gen':>4} {'max|b - mirror(b)|':>20} {'rms b':>12} {'asym / rms':>12}")
for g in gens:
    a = load(g, LIVE["policy out (dense_4)"])
    if a is None:
        continue
    asym = np.abs(a - a[::-1]).max()
    rms = np.sqrt((a ** 2).mean())
    print(f"{g:>4} {asym:>20.3e} {rms:>12.3e} "
          f"{(asym / rms if rms else float('nan')):>12.3f}")
