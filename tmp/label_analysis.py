"""Pull the value label apart: target = 0.8*z + 0.2*q.

z is recoverable: a decided game's last stored ply is the position the winning
move was played FROM, so its mover won -> z alternates back from +1. A drawn game
has z = 0 everywhere. Then q = (target - 0.8z)/0.2 exactly. If the reconstruction
is right, every recovered q lands in [-1, 1] -- that is the built-in check.

Then ask, on HELD-OUT data (model N vs gen N):
  1. is q adding signal or noise to the label?
  2. what does the net actually fit -- z, or q?
  3. is it calibrated, or is it hedging toward 0?
  4. how much of the residual is irreducible (repeat-position variance)?
"""
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import h5py as h5
import onnxruntime as rt

rt.set_default_logger_severity(4)

GEN = 18
QW = 0.2

fp = f"Connect4/Grok_Zero_Train/{GEN}/Self_Play_Data.h5"
with h5.File(fp, "r") as f:
    n = int(f["game_stats"][2])
    aug = ((len(f.keys()) - 1) // 3) // n
    X, T, Z, P, KEY = [], [], [], [], []
    for gid in range(n):
        brd = f[f"boards_{gid * aug}"][:]
        val = f[f"values_{gid * aug}"][:][:, 0]
        L = len(brd)
        decided = np.abs(val).max() > 0.25
        for ply in range(L):
            z = ((-1.0) ** (L - 1 - ply)) if decided else 0.0
            X.append(brd[ply])
            T.append(float(val[ply]))
            Z.append(z)
            P.append(ply)
            KEY.append(brd[ply].tobytes())

X = np.asarray(X, dtype=np.float32)
T, Z, P = np.asarray(T), np.asarray(Z), np.asarray(P)
Q = (T - (1 - QW) * Z) / QW

print(f"gen {GEN}: {n} games, {len(T)} positions")
inrange = np.mean((Q >= -1.001) & (Q <= 1.001))
print(f"reconstruction check: {inrange:.2%} of recovered q in [-1,1] "
      f"(min {Q.min():+.3f}, max {Q.max():+.3f})")
if inrange < 0.99:
    print("  !! reconstruction suspect, treat q numbers below with caution")

print()
print("1. IS q ADDING SIGNAL OR NOISE?")
dec = Z != 0
print(f"   corr(q, z) on decided positions : {np.corrcoef(Q[dec], Z[dec])[0, 1]:+.3f}")
print(f"   q sign agrees with z            : {np.mean(np.sign(Q[dec]) == Z[dec]):.1%}")
print(f"   MSE(q, z)                       : {np.mean((Q[dec] - Z[dec]) ** 2):.3f}")
print(f"   Var(z)              = {Z.var():.3f}")
print(f"   Var(0.8z)           = {(0.8 * Z).var():.3f}")
print(f"   Var(0.8z + 0.2q)    = {T.var():.3f}   <- the label you actually train on")
print(f"   Var(q)              = {Q.var():.3f}")

print()
print("2. WHAT DOES THE NET FIT?  (model 18 on gen 18 = held out)")
sess = rt.InferenceSession(f"Connect4/Grok_Zero_Train/{GEN}/model.onnx",
                           providers=["CPUExecutionProvider"])
V = []
for i in range(0, len(X), 1024):
    _, v = sess.run(None, {"inputs": X[i:i + 1024]})
    V.append(v[:, 0])
V = np.concatenate(V)
for name, tgt in (("z (true outcome)", Z), ("q (search value)", Q),
                  ("target 0.8z+0.2q", T)):
    print(f"   MSE(net, {name:<18}) = {np.mean((V - tgt) ** 2):.3f}    "
          f"corr = {np.corrcoef(V, tgt)[0, 1]:+.3f}")
print(f"   net output: mean {V.mean():+.3f}, std {V.std():.3f}   "
      f"| label std {T.std():.3f}, z std {Z.std():.3f}")
print("   (net std << label std means it is hedging toward 0)")

print()
print("3. CALIBRATION  -- when the net says X, what actually happens?")
print(f"   {'net V bin':>14} {'n':>7} {'mean net V':>11} {'mean z':>9} {'mean q':>9}")
edges = [-1.01, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 1.01]
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (V >= lo) & (V < hi)
    if m.sum() < 20:
        continue
    print(f"   [{lo:+.2f},{hi:+.2f}) {int(m.sum()):>7} {V[m].mean():>11.3f} "
          f"{Z[m].mean():>9.3f} {Q[m].mean():>9.3f}")

print()
print("4. HOW MUCH IS IRREDUCIBLE?  repeated positions, variance of z across games")
byk = defaultdict(list)
for k, z, p in zip(KEY, Z, P):
    byk[k].append(z)
buckets = [(0, 9, "0-9"), (10, 21, "10-21"), (22, 41, "22-41")]
plyof = {}
for k, p in zip(KEY, P):
    plyof.setdefault(k, p)
print(f"   {'ply':>8} {'n pos>=10':>10} {'mean Var(z)':>12} {'net MSE vs z':>13}")
for lo, hi, name in buckets:
    vs = [np.var(v) for k, v in byk.items() if len(v) >= 10 and lo <= plyof[k] <= hi]
    m = (P >= lo) & (P <= hi)
    print(f"   {name:>8} {len(vs):>10} "
          f"{(np.mean(vs) if vs else float('nan')):>12.3f} "
          f"{np.mean((V[m] - Z[m]) ** 2):>13.3f}")

print()
print("5. WOULD A DIFFERENT MIX BE EASIER TO FIT? label variance by q weight")
print(f"   {'q weight':>10} {'Var(label)':>12} {'corr(label,z)':>15}")
for qw in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
    lab = (1 - qw) * Z + qw * Q
    print(f"   {qw:>10.1f} {lab.var():>12.3f} {np.corrcoef(lab, Z)[0, 1]:>15.3f}")
