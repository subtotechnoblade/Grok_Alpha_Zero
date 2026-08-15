"""Explicit CPU vs DirectML on the run-3 net, repeated, because the iGPU numbers
were noisy enough (400-592 it/s for the SAME model) that one sample proves nothing.
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

MODEL = "Connect4/Grok_Zero_Train/6/model.onnx"
ITERS, REPS = 600, 3

print(f"available providers: {rt.get_available_providers()}\n")
print(f"run-3 net, {ITERS} iters x {REPS} reps\n")
print(f"{'provider':>16} {'batch':>6} " + "".join(f"{'rep'+str(i+1):>11}" for i in range(REPS)) + f"{'median':>11}")

for provider in ("CPUExecutionProvider", "DmlExecutionProvider"):
    if provider not in rt.get_available_providers():
        continue
    for bs in (1, 10):
        vals = []
        for _ in range(REPS):
            sess = rt.InferenceSession(MODEL, providers=[provider])
            x = np.random.uniform(-1, 2, (bs, 6, 7, 4)).astype(np.float32)
            for _ in range(300):
                sess.run(["policy", "value"], {"inputs": x})
            time.sleep(0.05)
            t0 = time.perf_counter()
            for _ in range(ITERS):
                sess.run(["policy", "value"], {"inputs": x})
            vals.append(ITERS / (time.perf_counter() - t0))
        print(f"{provider.replace('ExecutionProvider',''):>16} {bs:>6} "
              + "".join(f"{v:>11.1f}" for v in vals)
              + f"{sorted(vals)[len(vals)//2]:>11.1f}")
