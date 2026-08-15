"""Does trt_fp16_enable actually help at the batch sizes the inference server uses?

MCTS asks for batch 1, but self play routes every worker through the shared-memory
server, which batches up to num_workers requests. So the operating point is somewhere
between 1 and num_workers - and fp16 only pays once the GPU is compute-bound rather
than latency-bound.

Usage (WSL): python tmp/trt_batch_sweep.py
"""
import os
import sys
import time

import numpy as np
import onnxruntime as rt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from Connect4.Connect4 import Connect4, train_config

MODEL = f"{ROOT}/tmp/quant_out/connect4_fp32.onnx"
if not os.path.exists(MODEL):
    import glob
    c = sorted(glob.glob(f"{ROOT}/Connect4/Grok_Zero_Train/*/model.onnx"))
    MODEL = c[-1]
WORKERS = train_config["num_workers"]
BATCHES = [1, 2, 4, 8, WORKERS, 16, 32]
so = rt.SessionOptions()
so.log_severity_level = 3
MAXB = max(BATCHES)


def session(fp16, tag):
    cache = f"{ROOT}/tmp/quant_out/sweep_{tag}"
    os.makedirs(cache, exist_ok=True)
    opts = {"trt_engine_cache_enable": True, "trt_engine_cache_path": cache,
            "trt_timing_cache_enable": True, "trt_timing_cache_path": cache,
            "trt_builder_optimization_level": 5,
            "trt_profile_min_shapes": "inputs:1x6x7x4",
            "trt_profile_opt_shapes": f"inputs:{WORKERS}x6x7x4",
            "trt_profile_max_shapes": f"inputs:{MAXB}x6x7x4"}
    if fp16:
        opts["trt_fp16_enable"] = True
    return rt.InferenceSession(MODEL, sess_options=so, providers=[
        ("TensorrtExecutionProvider", opts), ("CUDAExecutionProvider", {}),
        "CPUExecutionProvider"])


def lat(sess, b, n=300):
    x = np.zeros((b, 6, 7, 4), dtype=np.float32)
    for _ in range(80):
        sess.run(["policy", "value"], {"inputs": x})
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(["policy", "value"], {"inputs": x})
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e6


print("=" * 76)
print(f"TRT fp32 vs fp16, same fp32 onnx | {os.path.basename(MODEL)} | eGPU pcie3 x4")
print(f"num_workers = {WORKERS}, so the server batches up to that")
s32, s16 = session(False, "fp32"), session(True, "fp16")
print("=" * 76)
print(f"{'batch':>6} {'fp32 us':>9} {'fp16 us':>9} {'speedup':>9} "
      f"{'fp32 st/s':>11} {'fp16 st/s':>11}")
for b in BATCHES:
    a, c = lat(s32, b), lat(s16, b)
    print(f"{b:>6} {a:>9.1f} {c:>9.1f} {a / c:>8.2f}x {b / a * 1e6:>11,.0f} "
          f"{b / c * 1e6:>11,.0f}")
print("=" * 76)
print("fp16 only pays once the GPU is compute-bound. At batch 1 the cost is kernel")
print("launch plus the pcie round trip, which half precision cannot shorten.")
