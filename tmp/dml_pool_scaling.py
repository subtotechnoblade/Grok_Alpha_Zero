"""How far does a pinned-session pool scale, and how does model size change that?

Builds N DirectML sessions (one per batch size 1..N) from the same free-dim model,
reporting incrementally with flush so a hard crash or OOM still shows where it died.

Usage:
  dml_pool_scaling.py [max_batch] [model_path]
    default model: tmp/dml_bench/model.onnx        (connect4, ~417k params)
    e.g. gomoku  : tmp/quant_out/gomoku_fp32.onnx  (~2.76M params, 15x15x2)
"""
import os
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAXB = int(sys.argv[1]) if len(sys.argv) > 1 else 32
MODEL = sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/tmp/dml_bench/model.onnx"

import numpy as np
import onnx
import onnxruntime as rt

try:
    import psutil
    proc = psutil.Process()
    rss = lambda: proc.memory_info().rss / 1024 ** 2
    avail = lambda: psutil.virtual_memory().available / 1024 ** 3
except ImportError:
    rss = avail = lambda: float("nan")

m = onnx.load(MODEL)
d = m.graph.input[0].type.tensor_type.shape.dim
DIM = d[0].dim_param
SHAPE = tuple(x.dim_value for x in d[1:])
OUTS = [o.name for o in m.graph.output]
n_param = sum(int(np.prod(t.dims)) for t in m.graph.initializer)

print(f"model      : {os.path.basename(MODEL)}")
print(f"params     : {n_param:,}   file {os.path.getsize(MODEL) / 1024**2:.1f} MB")
print(f"input      : batch({DIM!r}) x {SHAPE}   outputs {OUTS}")
print(f"target pool: {MAXB} sessions")
print(f"baseline   : RSS {rss():.0f} MB, system available {avail():.1f} GB", flush=True)


def make(b):
    so = rt.SessionOptions()
    so.log_severity_level = 3
    so.enable_mem_pattern = False
    so.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    so.add_free_dimension_override_by_name(DIM, b)
    return rt.InferenceSession(MODEL, sess_options=so,
                               providers=["DmlExecutionProvider", "CPUExecutionProvider"])


pool = {}
t0 = time.perf_counter()
MARKS = {1, 2, 4, 8, 16, 32, 48, 64, 96, 128, MAXB}
for b in range(1, MAXB + 1):
    ts = time.perf_counter()
    try:
        pool[b] = make(b)
    except Exception as e:
        print(f"  STOPPED at batch {b}: {type(e).__name__}: {str(e)[:80]}", flush=True)
        break
    if b in MARKS:
        print(f"  {b:>4} sessions  (+{time.perf_counter() - ts:5.2f}s)  "
              f"RSS {rss():>7.0f} MB  avail {avail():>5.1f} GB  "
              f"total {time.perf_counter() - t0:6.1f}s", flush=True)

build = time.perf_counter() - t0
n = len(pool)
print(f"\n{n}/{MAXB} sessions built in {build:.1f}s, RSS {rss():.0f} MB "
      f"({(rss() - 78) / max(1, n):.1f} MB per session)", flush=True)

print("\nverifying every session still runs:", flush=True)
bad = []
for b in sorted(pool):
    try:
        pool[b].run(OUTS, {"inputs": np.zeros((b, *SHAPE), dtype=np.float32)})
    except Exception as e:
        bad.append(f"{b}:{type(e).__name__}")
print(f"  {n - len(bad)}/{n} OK" + (f", problems: {bad[:6]}" if bad else ""))

print("\nlatency with the whole pool resident:", flush=True)
for b in [x for x in (1, 8, 32, 64, 128) if x in pool]:
    x = np.zeros((b, *SHAPE), dtype=np.float32)
    for _ in range(60):
        pool[b].run(OUTS, {"inputs": x})
    ts = []
    for _ in range(80):
        t = time.perf_counter()
        pool[b].run(OUTS, {"inputs": x})
        ts.append(time.perf_counter() - t)
    us = float(np.median(ts)) * 1e6
    print(f"  batch {b:>4}: {us:9.1f} us  {b / us * 1e6:9,.0f} states/s", flush=True)
