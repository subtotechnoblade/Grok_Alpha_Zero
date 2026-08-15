"""(1) Policy_Loss direct weight test with non-constant errors.
(2) Stage 6: Cache_Wrapper keyed on full input bytes -> side-to-move is part
    of the key (plane 0), so two states differing only in mover cannot collide.
"""
import sys, os
sys.path.insert(0, r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np

# ---------------------------------------------------------------- Policy_Loss
import tensorflow as tf
from Net.Alpha_Loss import Policy_Loss

y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1 / 3, 1 / 3, 1 / 3]], dtype=np.float32)
y_pred = np.array([[.8, .1, .1], [.1, .8, .1], [.5, .4, .1], [.2, .3, .5]], dtype=np.float32)
w = np.array([4.0, 2.0, 1.0, 0.0], dtype=np.float32)
ce = -(y_true * np.log(y_pred)).sum(1)
H_ps = float(np.sum(w * ce) / len(ce))
H_bc = float(np.mean(ce) * np.mean(w))
got = float(Policy_Loss(dtype="float32")(y_true, y_pred, sample_weight=w))
print("Policy_Loss direct test:")
print(f"  per-sample CEs {np.round(ce,4).tolist()}, weights {w.tolist()}")
print(f"  H_per_sample={H_ps:.6f}  H_broadcast={H_bc:.6f}  Policy_Loss={got:.6f}")
print(f"  -> {'PER-SAMPLE' if abs(got-H_ps)<1e-5 else 'BROADCAST (weights are a no-op per-sample)' if abs(got-H_bc)<1e-5 else 'NEITHER'}")

# ---------------------------------------------------------------- Cache
print()
print("Cache_Wrapper side-to-move key test:")
from Session_Cache import Cache_Wrapper

class MoverSensitiveSession:
    """value depends ONLY on side-to-move: +1 if plane0==-1 (i.e. +1 to... ) etc."""
    def run(self, output_names, input_feed):
        x = input_feed["inputs"][0]
        plane0 = float(x[0, 0, 0])
        v = 0.5 if plane0 == -1.0 else -0.5
        return [np.ones((1, 7), np.float32) / 7, np.array([[v]], np.float32)]

cw = Cache_Wrapper(MoverSensitiveSession(), max_cache_depth=10)

board = np.zeros((6, 7, 4), dtype=np.float32)
board[..., 3] = 0.0
a = board.copy(); a[:, :, 0] = -1.0   # -1 just moved
b = board.copy(); b[:, :, 0] = +1.0   # +1 just moved  (same position otherwise)

_, va = cw.run(["policy", "value"], {"inputs": a[None]}, depth=0)
_, vb = cw.run(["policy", "value"], {"inputs": b[None]}, depth=0)
_, va2 = cw.run(["policy", "value"], {"inputs": a[None]}, depth=0)  # should hit
s = cw.stats()
ok = (va[0][0] != vb[0][0]) and (va2[0][0] == va[0][0]) and s["entries"] == 2 and s["hits"] == 1
print(f"  value(A)={va[0][0]} value(B)={vb[0][0]} value(A again)={va2[0][0]}")
print(f"  stats: {s}")
print(f"  -> {'PASS: mover is part of the key, no collision' if ok else 'FAIL'}")

# depth gate: depth >= max_cache_depth bypasses the memo entirely
cw2 = Cache_Wrapper(MoverSensitiveSession(), max_cache_depth=2)
cw2.run(["policy", "value"], {"inputs": a[None]}, depth=5)
cw2.run(["policy", "value"], {"inputs": a[None]}, depth=5)
print(f"  depth>=max gate: entries={cw2.stats()['entries']} (expect 0), "
      f"hits={cw2.stats()['hits']} (expect 0)")
