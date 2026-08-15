"""Make the int8 QDQ model TRT-loadable. Diagnose, then fix, then PROVE it loads.

Attempt 1: exclude Reshape/Transpose nodes from quantization (stops QDQ propagation)
Attempt 2: inject value_info for every *_q_to_dq tensor, copying shapes from the
           pre-processed model (which quant_pre_process fully shape-inferred)
"""
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper

ROOT = "/mnt/c/Users/TryHarder/Desktop/Pythoncode/Grok_Zero"
sys.path.insert(0, ROOT + "/tmp")
OUT = f"{ROOT}/tmp/quant_out"

from quantize_int8 import H5Reader, load_states, QuantFormat, QuantType, CalibrationMethod
from onnxruntime.quantization import quantize_static
import onnxruntime as rt

game = sys.argv[1] if len(sys.argv) > 1 else "connect4"
folder = {"connect4": "Connect4", "gomoku": "Gomoku"}[game]
dims = {"connect4": "6x7x4", "gomoku": "15x15x2"}[game]
PRE = f"{OUT}/{game}_pre.onnx"
INT8_TRT = f"{OUT}/{game}_int8_trt.onnx"

states = load_states(f"{ROOT}/{folder}", 3, 1536, np.random.default_rng(0))
calib = states[:1024]

pre_model = onnx.load(PRE)
g = pre_model.graph
value_nodes = [n.name for n in g.node
               if n.op_type in ("MatMul", "Gemm") and "value" in n.name.lower()]
shape_ops = [n.name for n in g.node if n.op_type in ("Reshape", "Transpose", "Squeeze")]

quantize_static(PRE, INT8_TRT, H5Reader(calib.copy()),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                per_channel=True,
                calibrate_method=CalibrationMethod.MinMax,
                op_types_to_quantize=["Conv", "MatMul", "Gemm"],
                nodes_to_exclude=sorted(set(value_nodes + shape_ops)),
                extra_options={
                    "ActivationSymmetric": True,   # zero-point 0, TRT requirement
                    "AddQDQPairToWeight": True,    # fp32 weight + Q/DQ pair, TRT folds it
                    "QuantizeBias": False,         # TRT: bias must stay float
                    "DedicatedQDQPair": True,
                })

m = onnx.load(INT8_TRT)
qdq_tensors = {t for n in m.graph.node for t in list(n.input) + list(n.output)
               if "_q_to_dq" in t}
print(f"attempt 1 (exclude shape ops): {len(qdq_tensors)} q_to_dq tensors remain")

if qdq_tensors:
    # attempt 2: give them shapes. X_q_to_dq is int8-typed, same dims as X in PRE.
    pre_shapes = {}
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        pre_shapes[vi.name] = vi.type.tensor_type.shape
    have = {vi.name for vi in m.graph.value_info}
    added = 0
    for t in qdq_tensors:
        if t in have:
            continue
        base = t[: t.index("_q_to_dq")]
        if base not in pre_shapes:
            print(f"  no source shape for {t} (base {base!r})")
            continue
        vi = helper.make_value_info(
            t, helper.make_tensor_type_proto(TensorProto.INT8, None))
        vi.type.tensor_type.shape.CopyFrom(pre_shapes[base])
        m.graph.value_info.append(vi)
        added += 1
    onnx.save(m, INT8_TRT)
    print(f"attempt 2: injected {added} value_info entries")

# the proof: does a TRT session actually initialize and run?
CACHE = f"{OUT}/trt_cache_{game}"
os.makedirs(CACHE, exist_ok=True)
so = rt.SessionOptions()
so.log_severity_level = 3
providers = [("TensorrtExecutionProvider", {
    "trt_int8_enable": True, "trt_fp16_enable": True,
    "trt_engine_cache_enable": True, "trt_engine_cache_path": CACHE,
    "trt_timing_cache_enable": True, "trt_timing_cache_path": CACHE,
    "trt_profile_min_shapes": f"inputs:1x{dims}",
    "trt_profile_opt_shapes": f"inputs:16x{dims}",
    "trt_profile_max_shapes": f"inputs:64x{dims}"}),
    ("CUDAExecutionProvider", {}), "CPUExecutionProvider"]
sess = rt.InferenceSession(INT8_TRT, sess_options=so, providers=providers)
p, v = sess.run(["policy", "value"], {"inputs": states[1024:1040]})
print(f"TRT int8 session OK: policy {p.shape}, value {v.shape}")
