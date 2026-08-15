"""Same two models, but with the DirectML device pinned explicitly.

config has dml_device_id = None, which means DirectML picks its default adapter.
If the default is not the adapter you think it is, a "slowdown" between runs can be
nothing but a device change.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import onnxruntime as rt

from Client_Server import preload_gpu_dlls
preload_gpu_dlls()
rt.set_default_logger_severity(3)

MODELS = [("run2 (old head)", "tmp/run2_archive_gen0-19_oldvaluehead/18/model.onnx"),
          ("run3 (new head)", "Connect4/Grok_Zero_Train/6/model.onnx")]
BATCH, ITERS = 10, 600


def bench(path, opts):
    sess = rt.InferenceSession(path, providers=[("DmlExecutionProvider", opts)])
    x = np.random.uniform(-1, 2, (BATCH, 6, 7, 4)).astype(np.float32)
    for _ in range(300):
        sess.run(["policy", "value"], {"inputs": x})
    time.sleep(0.05)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        sess.run(["policy", "value"], {"inputs": x})
    return ITERS / (time.perf_counter() - t0)


print(f"batch {BATCH}, {ITERS} iters, DmlExecutionProvider\n")
print(f"{'device':>22} " + "".join(f"{n:>22}" for n, _ in MODELS))
rows = {}
for dev in ("default", 0, 1):
    opts = {} if dev == "default" else {"device_id": dev}
    cells, vals = "", []
    for name, path in MODELS:
        try:
            v = bench(path, opts)
            vals.append(v)
            cells += f"{v:>15.1f} it/s"
        except Exception as e:
            cells += f"{type(e).__name__:>22}"
    rows[dev] = vals
    print(f"{str(dev):>22} {cells}")

print()
if 0 in rows and 1 in rows and len(rows[0]) and len(rows[1]):
    print(f"device 1 vs device 0, run3 model: {rows[1][1] / rows[0][1]:.2f}x")
print("memory note from earlier sessions: DML device 0 = AMD 760M iGPU, "
      "device 1 = RTX 2060")
print("user reports ~478-500 it/s on the iGPU now, ~700-800 on the previous run")
