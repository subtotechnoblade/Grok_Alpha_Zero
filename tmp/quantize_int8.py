"""INT8 static quantization test for the Grok_Zero models. NOT integrated - measurement only.

Quantizes the latest connect4 + gomoku ONNX models to QDQ INT8, calibrated on REAL
self-play states from the h5 files, then measures on CPUExecutionProvider:
  - latency/throughput fp32 vs int8, batch 1 and 16 (single thread, nice-d, so the
    live training run is disturbed as little as possible)
  - fidelity: policy top-1 agreement, KL(fp32 || int8), value MAE/max on 512 states
  - file size

Outputs land in tmp/quant_out/, source models are never touched.

Usage (WSL): nice -n 19 python tmp/quantize_int8.py
"""
import os
import sys
import time
import glob

import numpy as np
import h5py
import onnxruntime as rt
from onnxruntime.quantization import (quantize_static, CalibrationDataReader,
                                      QuantFormat, QuantType, CalibrationMethod)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/tmp/quant_out"
os.makedirs(OUT, exist_ok=True)


def latest_gen(game_dir):
    gens = [int(os.path.basename(os.path.dirname(p)))
            for p in glob.glob(f"{game_dir}/Grok_Zero_Train/*/model.onnx")]
    return max(gens)


def load_states(game_dir, gens_back, n_states, rng):
    """Real board states across a few generations, network-input dtype float32."""
    latest = latest_gen(game_dir)
    states = []
    for g in range(max(0, latest - gens_back), latest):        # completed gens only
        path = f"{game_dir}/Grok_Zero_Train/{g}/Self_Play_Data.h5"
        if not os.path.exists(path):
            continue
        try:
            with h5py.File(path, "r") as f:
                n = (len(f.keys()) - 1) // 3
                for i in rng.permutation(n)[:40]:
                    states.append(f[f"boards_{i}"][:].astype(np.float32))
        except OSError:
            continue                                            # file busy (live writer)
    all_states = np.concatenate(states)
    return all_states[rng.permutation(len(all_states))[:n_states]]


class H5Reader(CalibrationDataReader):
    """Feeds real positions, one batch of 32 at a time."""
    def __init__(self, states, batch=32):
        self.batches = [states[i:i + batch] for i in range(0, len(states), batch)]
        self.i = 0

    def get_next(self):
        if self.i >= len(self.batches):
            return None
        feed = {"inputs": self.batches[self.i]}
        self.i += 1
        return feed


def bench(path, states, batch, iters=300, warmup=30):
    opts = rt.SessionOptions()
    opts.intra_op_num_threads = 1          # fair + minimal disturbance to the live run
    opts.log_severity_level = 3
    sess = rt.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
    feed = {"inputs": states[:batch]}
    for _ in range(warmup):
        sess.run(["policy", "value"], feed)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(["policy", "value"], feed)
        times.append(time.perf_counter() - t0)
    t = np.median(times)
    return t * 1000, batch / t              # ms/call, states/s


def fidelity(fp32_path, int8_path, states):
    opts = rt.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.log_severity_level = 3
    a = rt.InferenceSession(fp32_path, sess_options=opts, providers=["CPUExecutionProvider"])
    b = rt.InferenceSession(int8_path, sess_options=opts, providers=["CPUExecutionProvider"])
    pa, va = a.run(["policy", "value"], {"inputs": states})
    pb, vb = b.run(["policy", "value"], {"inputs": states})
    pa64, pb64 = pa.astype(np.float64) + 1e-12, pb.astype(np.float64) + 1e-12
    kl = float(np.mean(np.sum(pa64 * np.log(pa64 / pb64), axis=1)))
    top1 = float(np.mean(pa.argmax(1) == pb.argmax(1)))
    return {"top1": top1, "kl": kl,
            "v_mae": float(np.abs(va - vb).mean()),
            "v_max": float(np.abs(va - vb).max())}


def strip_float64(src, dst):
    """Remove Cast-to-double around the softmax (gomoku Build_Model.py:61 legacy).

    The fp64 policy output breaks quantize_static outright ("'policy' is not" an
    existing tensor) and blocks TensorRT. Verified lossless: max output diff 1e-7.
    """
    import onnx
    from onnx import TensorProto
    m = onnx.load(src)
    g = m.graph
    casts = [n for n in g.node if n.op_type == "Cast"
             and any(a.name == "to" and a.i == TensorProto.DOUBLE for a in n.attribute)]
    for cast in casts:
        s, d = cast.input[0], cast.output[0]
        for n in g.node:
            for i, inp in enumerate(n.input):
                if inp == d:
                    n.input[i] = s
        for out in g.output:
            if out.name == d:
                for n in g.node:
                    for i, o in enumerate(n.output):
                        if o == s:
                            n.output[i] = d
    names = {c.name for c in casts}
    keep = [n for n in g.node if n.name not in names]
    del g.node[:]
    g.node.extend(keep)
    for out in g.output:
        if out.type.tensor_type.elem_type == TensorProto.DOUBLE:
            out.type.tensor_type.elem_type = TensorProto.FLOAT
    del g.value_info[:]
    onnx.save(m, dst)
    return len(casts)


def run_game(name, game_dir, gens_back=3, n_calib=1024, n_eval=512):
    print("=" * 74)
    gen = latest_gen(game_dir)
    fp32 = f"{OUT}/{name}_fp32.onnx"
    pre = f"{OUT}/{name}_pre.onnx"
    int8 = f"{OUT}/{name}_int8.onnx"
    # pin a snapshot: the training run may advance latest_gen mid-measurement,
    # and int8-from-gen-N vs fp32-from-gen-N+1 produces nonsense fidelity numbers
    n_casts = strip_float64(f"{game_dir}/Grok_Zero_Train/{gen}/model.onnx", fp32)
    print(f"{name}: generation {gen} (pinned copy; {n_casts} fp64 cast(s) stripped)")

    rng = np.random.default_rng(0)
    states = load_states(game_dir, gens_back, n_calib + n_eval, rng)
    calib, evalst = states[:n_calib], states[n_calib:n_calib + n_eval]
    print(f"  {len(calib)} calibration states, {len(evalst)} eval states, "
          f"input {states.shape[1:]}")

    # shape inference + graph cleanup first; quantize_static wants this
    from onnxruntime.quantization.shape_inference import quant_pre_process
    quant_pre_process(fp32, pre, skip_symbolic_shape=False)

    # THE RECIPE (measured, see tmp/quant_out notes):
    # - only Conv/MatMul/Gemm. NEVER let the default op set quantize the softmax:
    #   at 225 classes the whole distribution rounds to one uint8 quantum and the
    #   policy comes out constant-uniform (measured: top-1 agreement 0.6%).
    # - value-head denses stay float: quantizing them tripled the value error.
    # - MinMax calibration beat Entropy on both games; u8 activations run ~1.6x
    #   faster than s8 on this CPU at identical fidelity.
    import onnx
    graph = onnx.load(pre).graph
    value_nodes = [n.name for n in graph.node
                   if n.op_type in ("MatMul", "Gemm") and "value" in n.name.lower()]
    quantize_static(pre, int8, H5Reader(calib),
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8,
                    per_channel=True,
                    calibrate_method=CalibrationMethod.MinMax,
                    op_types_to_quantize=["Conv", "MatMul", "Gemm"],
                    nodes_to_exclude=value_nodes)

    mb = lambda p: os.path.getsize(p) / 1024 ** 2
    print(f"  size: fp32 {mb(fp32):.2f} MB -> int8 {mb(int8):.2f} MB")

    print(f"  {'batch':>6} {'fp32 ms':>9} {'int8 ms':>9} {'speedup':>8} "
          f"{'fp32 st/s':>10} {'int8 st/s':>10}")
    for batch in (1, 16):
        f_ms, f_tp = bench(fp32, evalst, batch)
        q_ms, q_tp = bench(int8, evalst, batch)
        print(f"  {batch:>6} {f_ms:>9.3f} {q_ms:>9.3f} {f_ms / q_ms:>7.2f}x "
              f"{f_tp:>10.0f} {q_tp:>10.0f}")

    fid = fidelity(fp32, int8, evalst)
    print(f"  fidelity on {len(evalst)} states: top-1 agreement {fid['top1'] * 100:.1f}%, "
          f"KL {fid['kl']:.5f}, value MAE {fid['v_mae']:.4f}, worst {fid['v_max']:.4f}")
    return fid


if __name__ == "__main__":
    run_game("connect4", f"{ROOT}/Connect4")
    run_game("gomoku", f"{ROOT}/Gomoku")
    print("=" * 74)
    print("done. quantized models in tmp/quant_out/ (not integrated anywhere)")
