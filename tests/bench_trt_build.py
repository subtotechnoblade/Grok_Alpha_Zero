"""TensorRT engine build time and inference latency by builder optimization level, with a cold and a warm timing cache.
python -m tests.bench_trt_build <model.onnx> [workers] [leaves_per_batch]
Every build goes to a fresh scratch folder; the warm build reuses the timing cache the cold build of that level wrote."""
import os
import shutil
import sys
import tempfile
import time

import numpy as np

from Gomoku.game import Gomoku

LEVELS = (3, 5)
BATCHES = (1, 8, 48, 96)


def build(model, scratch, level, timing_dir, max_rows, shape):
    import onnxruntime as rt
    dims = "x".join(str(d) for d in shape)
    engine_dir = tempfile.mkdtemp(dir=scratch)
    options = {"trt_engine_cache_enable": True,
               "trt_dump_ep_context_model": True,
               "trt_builder_optimization_level": level,
               "trt_auxiliary_streams": 0,
               "trt_timing_cache_enable": True,
               "trt_timing_cache_path": timing_dir + "/",
               "trt_fp16_enable": True,
               "trt_ep_context_file_path": engine_dir + "/",
               "trt_profile_min_shapes": f"inputs:1x{dims}",
               "trt_profile_max_shapes": f"inputs:{max_rows}x{dims}",
               "trt_profile_opt_shapes": f"inputs:{max_rows}x{dims}"}
    t0 = time.perf_counter()
    sess = rt.InferenceSession(model, providers=[("TensorrtExecutionProvider", options), "CPUExecutionProvider"])
    sess.run(["policy", "value"], {"inputs": np.zeros((1, *shape), np.float32)})
    return sess, time.perf_counter() - t0


def latency(sess, shape, batch, iterations=400):
    x = np.random.uniform(-1, 1, (batch, *shape)).astype(np.float32)
    for _ in range(50):
        sess.run(["policy", "value"], {"inputs": x})
    t0 = time.perf_counter()
    for _ in range(iterations):
        sess.run(["policy", "value"], {"inputs": x})
    return (time.perf_counter() - t0) / iterations


if __name__ == "__main__":
    model = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    max_rows = workers * k
    shape = Gomoku().get_input_state().shape
    scratch = tempfile.mkdtemp(prefix="trt_build_bench_")
    try:
        for level in LEVELS:
            timing_dir = tempfile.mkdtemp(dir=scratch)
            for cache in ("cold", "warm"):
                sess, seconds = build(model, scratch, level, timing_dir, max_rows, shape)
                ms = [1e3 * latency(sess, shape, b) for b in BATCHES if b <= max_rows]
                print(f"level {level} {cache} timing cache: build {seconds:6.1f}s, ms/call "
                      + "  ".join(f"b{b} {m:.3f}" for b, m in zip(BATCHES, ms)), flush=True)
                del sess
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
