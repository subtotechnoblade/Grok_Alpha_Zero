"""Keep the free batch dim in the model, pin it per SESSION.

add_free_dimension_override_by_name() is a SessionOptions setting, so the .onnx on
disk keeps dim_param="unk__130" and every consumer (TensorRT, CPU EP, quantization)
still sees a dynamic batch. Only the DML session pins it.

That means a POOL: one session per batch size the server can send, all built from
the same file. This script validates the pool (DML can access-violate when several
sessions coexist) and measures it against a single free-dim session.

Run:
  C:\\Users\\TryHarder\\anaconda3\\envs\\tensorflow_updated\\python.exe tmp/dml_session_pool.py
"""
import os
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import onnx
import onnxruntime as rt

MODEL = f"{ROOT}/tmp/dml_bench/model.onnx"
MAX_BATCH = 6                      # = num_workers after Self_Play's cpu_count()//2 clamp


def batch_dim_name(model_path):
    """Read the free dimension's name from the model instead of hardcoding it.

    tf2onnx generates names like 'unk__130' which change on every re-export, so
    hardcoding one would silently stop overriding after the next Create_onnx.
    """
    m = onnx.load(model_path)
    d0 = m.graph.input[0].type.tensor_type.shape.dim[0]
    return d0.dim_param or None


def make_session(model_path, batch=None, dim_name=None):
    so = rt.SessionOptions()
    so.log_severity_level = 3
    so.enable_mem_pattern = False                        # required by the DML EP
    so.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL  # required by the DML EP
    if batch is not None and dim_name:
        so.add_free_dimension_override_by_name(dim_name, batch)
    return rt.InferenceSession(model_path, sess_options=so,
                               providers=["DmlExecutionProvider", "CPUExecutionProvider"])


name = batch_dim_name(MODEL)
print(f"model        : {os.path.basename(MODEL)}")
print(f"free dim name: {name!r}   (read at runtime, not hardcoded)")
print(f"file on disk : unchanged, still dynamic batch")
print()

# ---------------------------------------------------------------- build the pool
t0 = time.perf_counter()
pool = {}
for b in range(1, MAX_BATCH + 1):
    pool[b] = make_session(MODEL, batch=b, dim_name=name)
build = time.perf_counter() - t0
free = make_session(MODEL)          # no override, free dim
print(f"built {len(pool)} pinned sessions + 1 free-dim session in {build:.1f}s "
      f"(no crash)")

try:
    import psutil
    print(f"process RSS with all sessions live: "
          f"{psutil.Process().memory_info().rss / 1024**2:.0f} MB")
except ImportError:
    pass
print()


def bench(sess, b, warmup=250, iters=250):
    x = np.zeros((b, 6, 7, 4), dtype=np.float32)
    for _ in range(warmup):
        sess.run(["policy", "value"], {"inputs": x})
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(["policy", "value"], {"inputs": x})
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e6


print(f"{'batch':>6} {'free-dim us':>12} {'pinned us':>11} {'speedup':>9} "
      f"{'pinned st/s':>12}")
for b in range(1, MAX_BATCH + 1):
    f, p = bench(free, b), bench(pool[b], b)
    print(f"{b:>6} {f:>12.1f} {p:>11.1f} {f / p:>8.2f}x {b / p * 1e6:>12,.0f}")

# ---------------------------------------------------------------- correctness
print()
rng = np.random.default_rng(0)
x = rng.integers(-1, 2, size=(MAX_BATCH, 6, 7, 4)).astype(np.float32)
pf, vf = free.run(["policy", "value"], {"inputs": x})
pp, vp = pool[MAX_BATCH].run(["policy", "value"], {"inputs": x})
print(f"outputs identical to the free-dim session: "
      f"policy max|diff| {np.abs(pf - pp).max():.2e}, "
      f"value max|diff| {np.abs(vf - vp).max():.2e}")

# a pinned session REJECTS other batch sizes - the server must route correctly
try:
    pool[MAX_BATCH].run(["policy", "value"], {"inputs": x[:1]})
    print("pinned session accepted a mismatched batch (unexpected)")
except Exception as e:
    print(f"pinned session correctly rejects a mismatched batch: {type(e).__name__}")
