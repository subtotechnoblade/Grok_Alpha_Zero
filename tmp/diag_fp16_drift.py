"""Does TRT fp16 still agree with fp32 as the weights drift across generations?

Self play evaluates positions through <gen>/TRT_cache/model_ctx.onnx (TensorRT,
fp16 allowed); training fits the fp32 weights. fp16 error is weight dependent:
if activations inflate over generations, the engine self play sees can drift
away from the net training thinks it deployed. Compare per generation on the
same positions: top-1 policy agreement + value MAE, TRT fp16 vs CPU fp32.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import onnxruntime as rt

rt.set_default_logger_severity(4)
from Client_Server import preload_gpu_dlls
preload_gpu_dlls()

from Connect4.game import Connect4

# ~400 positions from random playouts, plus the empty board first
rng = np.random.default_rng(0)
states = []
g0 = Connect4()
states.append(g0.get_input_state().copy())
while len(states) < 400:
    g = Connect4()
    winner = -2
    while winner == -2:
        la = g.get_legal_actions()
        g.do_action(la[rng.integers(len(la))])
        winner = g.check_win()
        states.append(g.get_input_state().copy())
        if len(states) >= 400:
            break
X = np.asarray(states, dtype=np.float32)

def trt_session(gen):
    folder = f"Connect4/Grok_Zero_Train/{gen}"
    return rt.InferenceSession(
        f"{folder}/TRT_cache/model_ctx.onnx",
        providers=[("TensorrtExecutionProvider", {
            "trt_engine_cache_enable": True,
            "trt_fp16_enable": True,
            "trt_ep_context_file_path": f"{folder}/TRT_cache/",
        }), "CUDAExecutionProvider", "CPUExecutionProvider"])

print(f"{'gen':>4} {'top1 agree':>11} {'policy MAE':>11} {'value MAE':>10} "
      f"{'value max':>10}  empty-board value fp32 vs fp16")
for gen in (2, 4, 6, 7, 8, 9):
    cpu = rt.InferenceSession(f"Connect4/Grok_Zero_Train/{gen}/model.onnx",
                              providers=["CPUExecutionProvider"])
    trt = trt_session(gen)
    P32, V32, P16, V16 = [], [], [], []
    B = 10  # engine profile was built for max batch = num_workers = 10
    for i in range(0, len(X), B):
        chunk = X[i:i + B]
        p, v = cpu.run(None, {"inputs": chunk}); P32.append(p); V32.append(v)
        p, v = trt.run(None, {"inputs": chunk}); P16.append(p); V16.append(v)
    P32, V32 = np.concatenate(P32), np.concatenate(V32).ravel()
    P16, V16 = np.concatenate(P16), np.concatenate(V16).ravel()
    agree = float(np.mean(P32.argmax(1) == P16.argmax(1)))
    pmae = float(np.abs(P32 - P16).mean())
    vmae = float(np.abs(V32 - V16).mean())
    vmax = float(np.abs(V32 - V16).max())
    print(f"{gen:>4} {agree:>10.1%} {pmae:>11.5f} {vmae:>10.5f} {vmax:>10.5f}"
          f"   {V32[0]:+.4f} vs {V16[0]:+.4f}")
    del trt, cpu
