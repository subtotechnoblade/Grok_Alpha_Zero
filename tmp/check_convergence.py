"""Snapshot the run: per-gen outcome split + each net's opening policy/value.

Read-only. Skips a gen whose h5 is mid-write.
"""
import os
import sys
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import h5py as h5
import onnxruntime as rt

rt.set_default_logger_severity(4)

from Connect4.game import Connect4

GENS = sorted(glob.glob("Connect4/Grok_Zero_Train/*"),
              key=lambda p: int(os.path.basename(p)))

print("1. OUTCOMES PER GENERATION (self-play data produced BY that gen's net)")
print(f"{'gen':>4} {'games':>6} {'p-1 win':>8} {'draw':>6} {'p+1 win':>8} {'avg mv':>7}")
for d in GENS:
    f = os.path.join(d, "Self_Play_Data.h5")
    if not os.path.exists(f):
        continue
    try:
        with h5.File(f, "r") as h:
            ma, ta, n, w1, dr, w2 = h["game_stats"][:]
        n = int(n)
        if n <= 0:
            continue
        print(f"{os.path.basename(d):>4} {n:>6} {int(w1)/n:>8.3f} {int(dr)/n:>6.3f} "
              f"{int(w2)/n:>8.3f} {int(ta)/n:>7.1f}")
    except OSError:
        print(f"{os.path.basename(d):>4}  (h5 busy/mid-write, skipped)")

print()
print("2. EACH NET'S OPENING BELIEF (empty board, p-1 to move)")
print(f"{'gen':>4} {'argmax':>7} {'P(col3)':>8} {'P(col1)':>8} {'V(start)':>9}   full policy")
g = Connect4()
state = g.get_input_state()[None].astype(np.float32)
for d in GENS:
    m = os.path.join(d, "model.onnx")
    if not os.path.exists(m):
        continue
    try:
        sess = rt.InferenceSession(m, providers=["CPUExecutionProvider"])
        pol, val = sess.run(None, {"inputs": state})
        p = pol[0]
        pol_str = " ".join(f"{x:.2f}" for x in p)
        print(f"{os.path.basename(d):>4} {int(np.argmax(p)):>7} {p[3]:>8.3f} "
              f"{p[1]:>8.3f} {float(val[0][0]):>+9.3f}   [{pol_str}]")
    except Exception as e:
        print(f"{os.path.basename(d):>4}  (load failed: {e})")

print()
print("3. VALUE ALONG THE CENTER LINE (latest net): start, after col3, after col3-col3")
latest = GENS[-1]
sess = rt.InferenceSession(os.path.join(latest, "model.onnx"),
                           providers=["CPUExecutionProvider"])
g = Connect4()
for label, action in (("empty board (p-1 moves)", None),
                      ("after 3 (p+1 moves)", 3),
                      ("after 3,3 (p-1 moves)", 3),
                      ("after 3,3,3 (p+1 moves)", 3)):
    if action is not None:
        g.do_action(action)
    s = g.get_input_state()[None].astype(np.float32)
    pol, val = sess.run(None, {"inputs": s})
    print(f"  {label:<26} V={float(val[0][0]):+.3f}  argmax={int(np.argmax(pol[0]))}")
