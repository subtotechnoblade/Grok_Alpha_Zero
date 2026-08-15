"""Head-to-head: run 2's exported net vs run 3's, same provider, same batch, same
process. Mirrors Compute_Speed.compute_speed (batch = num_workers = 10, 500 warmup).

Also diffs the ONNX graphs, because at batch 10 on a small net the cost is dominated
by the NUMBER of dispatched operators, not by FLOPs.
"""
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import onnx
import onnxruntime as rt

from Client_Server import preload_gpu_dlls
preload_gpu_dlls()
rt.set_default_logger_severity(3)

A = ("run2 (old head)", "tmp/run2_archive_gen0-19_oldvaluehead/18/model.onnx")
B = ("run3 (new head)", "Connect4/Grok_Zero_Train/6/model.onnx")
BATCHES = [1, 10, 32]
ITERS = 600

print("=== ONNX GRAPH DIFF ===")
counts = {}
for label, path in (A, B):
    m = onnx.load(path)
    c = Counter(n.op_type for n in m.graph.node)
    counts[label] = c
    print(f"{label:<18} {len(m.graph.node):>4} nodes   {dict(sorted(c.items()))}")

ka, kb = counts[A[0]], counts[B[0]]
print("\nper-op delta (run3 - run2), non-zero only:")
for op in sorted(set(ka) | set(kb)):
    d = kb.get(op, 0) - ka.get(op, 0)
    if d:
        print(f"   {op:<16} {ka.get(op, 0):>3} -> {kb.get(op, 0):>3}   ({d:+d})")

avail = rt.get_available_providers()
prov = ("DmlExecutionProvider" if "DmlExecutionProvider" in avail
        else "CPUExecutionProvider")
print(f"\n=== SPEED ({prov}) ===")
if prov == "CPUExecutionProvider":
    print("  !! DirectML not available in this env - numbers are CPU, not comparable "
          "to the pipeline's GPU figures")

print(f"{'batch':>6} " + "".join(f"{lbl:>22}" for lbl, _ in (A, B)) + f"{'ratio':>9}")
for bs in BATCHES:
    row, res = f"{bs:>6} ", []
    for label, path in (A, B):
        sess = rt.InferenceSession(path, providers=[prov])
        shape = sess.get_inputs()[0].shape
        x = np.random.uniform(-1, 2, (bs, *[d if isinstance(d, int) else 6
                                            for d in shape[1:]])).astype(np.float32)
        for _ in range(300):
            sess.run(["policy", "value"], {"inputs": x})
        time.sleep(0.05)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            sess.run(["policy", "value"], {"inputs": x})
        dt = time.perf_counter() - t0
        ips = ITERS / dt
        res.append(ips)
        row += f"{ips:>15.1f} it/s"
    row += f"{res[1] / res[0]:>9.2f}x"
    print(row)

print("\nratio < 1.00 means run 3 is SLOWER than run 2 at that batch size")
