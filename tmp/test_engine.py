"""Tests for the Client_Server inference engine abstraction.

  1. bucket_for logic, both engines, no GPU needed
  2. BucketedEngine builds and every session runs
  3. padded rows do not change the real rows' outputs (the correctness crux)
  4. DynamicEngine is a no-op passthrough, so CPU/CUDA/TRT behave as before

Run:
  C:\\Users\\TryHarder\\anaconda3\\envs\\tensorflow_updated\\python.exe tmp/test_engine.py
"""
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import onnx
from Client_Server import make_engine, DynamicEngine, BucketedEngine, _needs_fixed_shapes

MODEL = f"{ROOT}/tmp/dml_bench/model.onnx"
DML = ["DmlExecutionProvider", "CPUExecutionProvider"]
CPU = ["CPUExecutionProvider"]

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        fails.append(name)


print("1. provider detection + factory")
check("Dml detected", _needs_fixed_shapes(DML))
check("plain CPU not detected", not _needs_fixed_shapes(CPU))
check("tuple-form providers detected",
      _needs_fixed_shapes([("DmlExecutionProvider", {}), "CPUExecutionProvider"]))
check("factory -> DynamicEngine for CPU",
      isinstance(make_engine(MODEL, CPU, 6), DynamicEngine))

print("\n2. bucket_for")
dyn = make_engine(MODEL, CPU, 6)
check("DynamicEngine passes batch through",
      all(dyn.bucket_for(b) == b for b in range(1, 40)))

buckets = BucketedEngine._default_buckets(6)
check("small max_batch -> every size", list(buckets) == [1, 2, 3, 4, 5, 6],
      str(list(buckets)))
big = sorted(set(BucketedEngine._default_buckets(32)))
check("large max_batch -> tight buckets, includes 1 and 32",
      big[0] == 1 and big[-1] == 32 and len(big) < 32, f"{len(big)} buckets: {big}")


class _FakeBucketed(BucketedEngine):
    def __init__(self, buckets):          # skip session building for pure logic tests
        self.buckets = sorted(set(buckets))
        self.max_batch = self.buckets[-1]


fb = _FakeBucketed([1, 2, 3, 4, 8, 12, 16])
check("rounds up to nearest bucket",
      [fb.bucket_for(n) for n in (1, 5, 9, 13, 16)] == [1, 8, 12, 16, 16],
      str([fb.bucket_for(n) for n in (1, 5, 9, 13, 16)]))
check("over-max clamps to max", fb.bucket_for(99) == 16)

print("\n3. BucketedEngine on DirectML")
try:
    eng = make_engine(MODEL, DML, 6)
    check("factory -> BucketedEngine for Dml", isinstance(eng, BucketedEngine))
    check("dim name read from model, not hardcoded",
          bool(eng.dim_name) and eng.dim_name ==
          onnx.load(MODEL).graph.input[0].type.tensor_type.shape.dim[0].dim_param,
          repr(eng.dim_name))
    check("one session per bucket", len(eng.sessions) == len(eng.buckets),
          f"{len(eng.sessions)} sessions")
    ok = True
    for b in eng.buckets:
        try:
            eng.run(["policy", "value"],
                    {"inputs": np.zeros((b, 6, 7, 4), np.float32)})
        except Exception as e:
            ok = False
            print(f"      batch {b} failed: {type(e).__name__}")
    check("every pinned session runs", ok)

    print("\n4. drop-in behaviour: any batch in, same batch out")
    # sparse buckets so padding is actually exercised
    sparse = BucketedEngine(MODEL, DML, 8, buckets=[1, 4, 8])
    rng = np.random.default_rng(0)
    shapes_ok, values_ok = True, True
    for n in range(1, 9):
        x = rng.integers(-1, 2, size=(n, 6, 7, 4)).astype(np.float32)
        p, v = sparse.run(["policy", "value"], {"inputs": x})
        if p.shape[0] != n or v.shape[0] != n:
            shapes_ok = False
            print(f"      batch {n} -> policy {p.shape}, value {v.shape}")
        # every row must match what an exact-size session computes for it
        pe, ve = eng.run(["policy", "value"], {"inputs": x[:min(n, 6)]})
        k = min(n, 6)
        if not (np.allclose(p[:k], pe, atol=1e-5) and np.allclose(v[:k], ve, atol=1e-5)):
            values_ok = False
    check("output batch always equals input batch", shapes_ok)
    check("padded rows never disturb the real ones", values_ok)
    check("run() takes the rt.InferenceSession signature",
          sparse.run(["policy"], {"inputs": np.zeros((3, 6, 7, 4), np.float32)},
                     None)[0].shape[0] == 3)

    print("\n5. DirectML vs CPU through the same interface")
    import time

    def lat(engine, b):
        x = np.zeros((b, 6, 7, 4), np.float32)
        for _ in range(200):
            engine.run(["policy", "value"], {"inputs": x})
        ts = []
        for _ in range(200):
            t0 = time.perf_counter()
            engine.run(["policy", "value"], {"inputs": x})
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts)) * 1e6

    cpu_eng = make_engine(MODEL, CPU, 6)
    for b in (1, 6):
        print(f"     batch {b}: bucketed DML {lat(eng, b):7.1f} us   "
              f"dynamic CPU {lat(cpu_eng, b):7.1f} us")
except Exception as e:
    check(f"DirectML engine construction ({type(e).__name__}: {str(e)[:50]})", False)

print()
print("ALL PASSED" if not fails else f"FAILURES: {fails}")
sys.exit(1 if fails else 0)
