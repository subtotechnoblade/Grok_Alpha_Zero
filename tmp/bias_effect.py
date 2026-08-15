"""Does the non-zero bias actually affect the output?

First: did tf2onnx already fold BN into the convs? If BatchNormalization nodes
survive, the Conv/Gemm biases in the graph are the ORIGINAL trained biases and
the fold question is still open. If they are gone, the biases we see are fused
(beta - gamma*mu/sigma) and are load-bearing by construction.

Then: zero every Conv/Gemm bias whose consumer is a BatchNormalization, re-run,
and compare outputs on real positions. If the claim holds, outputs are identical.
"""
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as rt
import h5py as h5

rt.set_default_logger_severity(4)

SRC = "Connect4/Grok_Zero_Train/18/model.onnx"
OUT = "tmp/model18_zeroed.onnx"

m = onnx.load(SRC)
types = Counter(n.op_type for n in m.graph.node)
print("ONNX node types:", dict(types))
has_bn = types.get("BatchNormalization", 0)
print(f"BatchNormalization nodes surviving export: {has_bn}")
print("  -> BN was NOT folded by tf2onnx" if has_bn else
      "  -> BN already folded into the convs at export time")
print()

# map: tensor name -> node that consumes it
consumers = {}
for n in m.graph.node:
    for i in n.input:
        consumers.setdefault(i, []).append(n)

inits = {i.name: i for i in m.graph.initializer}
zeroed = []
for n in m.graph.node:
    if n.op_type not in ("Conv", "Gemm", "MatMul") or len(n.input) < 3:
        continue
    bias_name = n.input[2]
    if bias_name not in inits:
        continue
    followed_by_bn = any(c.op_type == "BatchNormalization"
                         for c in consumers.get(n.output[0], []))
    arr = numpy_helper.to_array(inits[bias_name])
    if followed_by_bn:
        inits[bias_name].CopyFrom(
            numpy_helper.from_array(np.zeros_like(arr), bias_name))
        zeroed.append((n.op_type, bias_name, float(np.abs(arr).max())))

print(f"zeroed {len(zeroed)} biases whose output feeds a BatchNormalization:")
for t, name, mx in zeroed:
    print(f"   {t:<7} {name[:52]:<52} max|b| was {mx:.3e}")

if not zeroed:
    print("   (none -- nothing to test)")
    sys.exit(0)

onnx.save(m, OUT)

# real positions to test on
with h5.File("Connect4/Grok_Zero_Train/18/Self_Play_Data.h5", "r") as f:
    n_games = int(f["game_stats"][2])
    aug = ((len(f.keys()) - 1) // 3) // n_games
    X = np.concatenate([f[f"boards_{g * aug}"][:] for g in range(0, 60)]).astype(np.float32)

a = rt.InferenceSession(SRC, providers=["CPUExecutionProvider"])
b = rt.InferenceSession(OUT, providers=["CPUExecutionProvider"])
pa, va = a.run(None, {"inputs": X})
pb, vb = b.run(None, {"inputs": X})

print()
print(f"compared on {len(X)} real gen-18 positions")
print(f"  value  max abs diff : {np.abs(va - vb).max():.3e}")
print(f"  policy max abs diff : {np.abs(pa - pb).max():.3e}")
print(f"  value  spread for scale reference: {va.min():+.3f} .. {va.max():+.3f}")
