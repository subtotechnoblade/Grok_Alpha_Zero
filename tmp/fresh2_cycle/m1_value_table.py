"""M1: VALUE-CONDEMNATION TABLE.

For each gen's model.onnx: net value of the position after the opener plays
column c. Sign convention verified empirically first with forced-win positions.

Convention hypothesis (from code): the net is trained on values_[ply] which are
from the perspective of the player who MOVES at that ply (the next mover of the
stored state). So for a 1-stone board (opener played c), the net's value is from
p+1's perspective, and opener-Q = -value.

Verification: build positions where the next mover has an immediate 4-in-a-row.
If value ~ +1, convention holds. Test both parities.
"""
import os, sys
ROOT = r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import onnxruntime as rt
rt.set_default_logger_severity(4)

from Connect4.game import Connect4

GENS = list(range(5, 13))

def state_after(actions):
    g = Connect4()
    for a in actions:
        g.do_action(a)
    return g.get_input_state()[None].astype(np.float32), g.get_next_player()

def net(sess, x):
    pol, val = sess.run(None, {"inputs": x})
    return pol[0], float(val[0][0])

# --- sign convention probes ---
# A: mover = -1 (opener side), has immediate win at col3 (bottom row 0,1,2)
probeA, npA = state_after([0, 6, 1, 6, 2, 5])
# B: mover = +1 (second player), has immediate win at col3 (bottom row 4,5,6)
probeB, npB = state_after([0, 4, 0, 5, 1, 6, 1])
assert npA == -1 and npB == 1

empty, _ = state_after([])

print("=== sign convention probes: value should be ~+1 for BOTH if 'next mover perspective' ===")
print(f"{'gen':>4} {'V(empty)':>9} {'V(A: -1 to move, wins)':>24} {'V(B: +1 to move, wins)':>24}")
sessions = {}
for gen in GENS:
    m = f"Connect4/Grok_Zero_Train/{gen}/model.onnx"
    if not os.path.exists(m):
        continue
    s = rt.InferenceSession(m, providers=["CPUExecutionProvider"])
    sessions[gen] = s
    _, ve = net(s, empty)
    _, va = net(s, probeA)
    _, vb = net(s, probeB)
    print(f"{gen:>4} {ve:>9.3f} {va:>24.3f} {vb:>24.3f}")

print()
print("=== opener-Q table: opener_Q(c) = -V(after opener plays c) ===")
print(f"{'gen':>4}  " + "  ".join(f"col{c:>4}" for c in range(7)) + "   argmaxQ  argmaxPrior")
rows = {}
for gen, s in sessions.items():
    qs = []
    for c in range(7):
        x, _ = state_after([c])
        _, v = net(s, x)
        qs.append(-v)
    qs = np.array(qs)
    pol, _ = net(s, empty)
    rows[gen] = (qs, pol)
    print(f"{gen:>4}  " + "  ".join(f"{q:>7.3f}" for q in qs)
          + f"   {int(np.argmax(qs)):>7} {int(np.argmax(pol)):>12}")

print()
print("=== raw prior at empty board per gen ===")
print(f"{'gen':>4}  " + "  ".join(f"col{c:>4}" for c in range(7)))
for gen, (qs, pol) in rows.items():
    print(f"{gen:>4}  " + "  ".join(f"{p:>7.3f}" for p in pol))

np.save(r"tmp\fresh2_cycle\opener_q.npy",
        np.array([[gen, *rows[gen][0], *rows[gen][1]] for gen in rows]))
