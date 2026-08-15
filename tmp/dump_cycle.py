"""Dump the saved arrays from the cycle agent so nothing is re-computed."""
import numpy as np

D = r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero\tmp\fresh2_cycle"
np.set_printoptions(precision=3, suppress=True, linewidth=200)

q = np.load(D + r"\opener_q.npy", allow_pickle=True)
print("=== opener_q.npy ===", getattr(q, "shape", None), getattr(q, "dtype", None))
print(q)

for name in ("data_stats.npz", "live_mcts.npz"):
    z = np.load(D + "\\" + name, allow_pickle=True)
    print(f"\n=== {name} ===")
    for k in z.files:
        a = z[k]
        print(f"\n-- {k}  shape={a.shape} dtype={a.dtype}")
        if a.dtype == object:
            print(a.item() if a.shape == () else a)
        else:
            print(a)
