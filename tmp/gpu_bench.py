"""GPU benchmark: CUDA fp32 vs TRT fp32 / fp16 / int8-QDQ on the pinned models.

TRT explicit quantization requires SYMMETRIC signed int8 Q/DQ (zero point 0), so a
QInt8-activation variant is built here; the uint8 one stays for CPU.
Usage: python gpu_bench.py {connect4|gomoku}
"""
import os
import sys
import gc
import time

import numpy as np
import onnx

ROOT = "/mnt/c/Users/TryHarder/Desktop/Pythoncode/Grok_Zero"
sys.path.insert(0, ROOT + "/tmp")
OUT = f"{ROOT}/tmp/quant_out"

from quantize_int8 import (H5Reader, load_states,
                           QuantFormat, QuantType, CalibrationMethod)
from onnxruntime.quantization import quantize_static
import onnxruntime as rt

game = sys.argv[1]
folder = {"connect4": "Connect4", "gomoku": "Gomoku"}[game]
dims = {"connect4": "6x7x4", "gomoku": "15x15x2"}[game]

FP32 = f"{OUT}/{game}_fp32.onnx"
PRE = f"{OUT}/{game}_pre.onnx"
INT8_TRT = f"{OUT}/{game}_int8_trt.onnx"

states = load_states(f"{ROOT}/{folder}", 3, 1536, np.random.default_rng(0))
calib, evalst = states[:1024], states[1024:1536]

# the TRT-loadable int8 model is built by fix_trt_int8.py (symmetric s8 activations,
# AddQDQPairToWeight, unquantized bias, DedicatedQDQPair) - proven to initialize
if not os.path.exists(INT8_TRT):
    raise SystemExit(f"missing {INT8_TRT}; run fix_trt_int8.py {game} first")

CACHE = f"{OUT}/trt_cache_{game}"
os.makedirs(CACHE, exist_ok=True)


def trt_opts(extra):
    o = {"trt_engine_cache_enable": True, "trt_engine_cache_path": CACHE,
         "trt_timing_cache_enable": True, "trt_timing_cache_path": CACHE,
         "trt_profile_min_shapes": f"inputs:1x{dims}",
         "trt_profile_opt_shapes": f"inputs:16x{dims}",
         "trt_profile_max_shapes": f"inputs:64x{dims}"}
    o.update(extra)
    return o


CONFIGS = [
    ("cuda_fp32", FP32, [("CUDAExecutionProvider", {}), "CPUExecutionProvider"]),
    ("trt_fp32", FP32, [("TensorrtExecutionProvider", trt_opts({})),
                        ("CUDAExecutionProvider", {}), "CPUExecutionProvider"]),
    ("trt_fp16", FP32, [("TensorrtExecutionProvider", trt_opts({"trt_fp16_enable": True})),
                        ("CUDAExecutionProvider", {}), "CPUExecutionProvider"]),
    ("trt_int8", INT8_TRT, [("TensorrtExecutionProvider",
                             trt_opts({"trt_int8_enable": True, "trt_fp16_enable": True})),
                            ("CUDAExecutionProvider", {}), "CPUExecutionProvider"]),
]


def bench_gpu(sess, batch, iters, warmup=100):
    feed = {"inputs": evalst[:batch]}
    for _ in range(warmup):
        sess.run(["policy", "value"], feed)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(["policy", "value"], feed)
        times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1000)


baseline = {}
print(f"\n{game}  (RTX 2060)")
print(f"{'config':>10} {'build s':>8} {'b1 ms':>8} {'b16 ms':>8} {'b64 ms':>8} "
      f"{'b16 st/s':>9} {'top-1':>7} {'KL':>9} {'vMAE':>8}")
for name, path, providers in CONFIGS:
    t0 = time.perf_counter()
    so = rt.SessionOptions()
    so.log_severity_level = 3
    sess = rt.InferenceSession(path, sess_options=so, providers=providers)
    sess.run(["policy", "value"], {"inputs": evalst[:16]})   # engine build happens here
    build = time.perf_counter() - t0

    b1 = bench_gpu(sess, 1, 400)
    b16 = bench_gpu(sess, 16, 300)
    b64 = bench_gpu(sess, 64, 200)

    # chunks of 64: the TRT optimization profile tops out there, one 512-batch
    # call fails setInputShape exactly like the num_workers change did in training
    ps, vs = [], []
    for i in range(0, len(evalst), 64):
        pc, vc = sess.run(["policy", "value"], {"inputs": evalst[i:i + 64]})
        ps.append(pc)
        vs.append(vc)
    p, v = np.concatenate(ps), np.concatenate(vs)
    if name == "cuda_fp32":
        baseline["p"], baseline["v"] = p, v
        top1, kl, mae = 1.0, 0.0, 0.0
    else:
        pa = baseline["p"].astype(np.float64) + 1e-12
        pb = p.astype(np.float64) + 1e-12
        kl = float(np.mean(np.sum(pa * np.log(pa / pb), axis=1)))
        top1 = float(np.mean(baseline["p"].argmax(1) == p.argmax(1)))
        mae = float(np.abs(baseline["v"] - v).mean())
    print(f"{name:>10} {build:>8.1f} {b1:>8.3f} {b16:>8.3f} {b64:>8.3f} "
          f"{16 / b16 * 1000:>9.0f} {top1 * 100:>6.1f}% {kl:>9.5f} {mae:>8.4f}")

    del sess
    gc.collect()
