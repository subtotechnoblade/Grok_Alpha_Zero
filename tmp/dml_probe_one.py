"""Benchmark ONE DirectML configuration in an isolated process.

DirectML can hard-crash (access violation) on some configurations, so each is run
as a separate subprocess by tmp/dml_tuning.py and a crash is reported, not fatal.

Usage: dml_probe_one.py <kind> <batch>
  kind: cpu | naive | required | fixed
"""
import os
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import onnx
import onnxruntime as rt

MODEL = f"{ROOT}/tmp/dml_bench/model.onnx"
kind, batch = sys.argv[1], int(sys.argv[2])
WARMUP, ITERS = 300, 300

so = rt.SessionOptions()
so.log_severity_level = 3
if kind in ("required", "fixed"):
    so.enable_mem_pattern = False                       # mandated by the DML docs
    so.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL  # mandated by the DML docs
if kind == "fixed":
    m = onnx.load(MODEL)
    d0 = m.graph.input[0].type.tensor_type.shape.dim[0]
    if d0.dim_param:
        so.add_free_dimension_override_by_name(d0.dim_param, batch)

providers = (["CPUExecutionProvider"] if kind == "cpu"
             else ["DmlExecutionProvider", "CPUExecutionProvider"])
sess = rt.InferenceSession(MODEL, sess_options=so, providers=providers)

x = np.zeros((batch, 6, 7, 4), dtype=np.float32)
for _ in range(WARMUP):
    sess.run(["policy", "value"], {"inputs": x})
ts = []
for _ in range(ITERS):
    t0 = time.perf_counter()
    sess.run(["policy", "value"], {"inputs": x})
    ts.append(time.perf_counter() - t0)
print(f"RESULT {kind} {batch} {float(np.median(ts)) * 1e6:.1f} {sess.get_providers()[0]}")
