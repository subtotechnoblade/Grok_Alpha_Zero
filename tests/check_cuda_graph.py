"""TensorRT CUDA graphs through IOBinding: one captured graph per padded bucket, checked against plain TensorRT.
python -m tests.check_cuda_graph <model_dir> <profile_max> [bucket,bucket,...]"""
import sys
import time

import numpy as np

from tests.mcts_scenarios import ROOT
from tests.bench_search_speed import providers_for, model_path
from Gomoku.game import Gomoku


class GraphEngine:
    def __init__(self, file_path, providers, buckets):
        import onnxruntime as rt
        from Client_Server import preload_gpu_dlls
        preload_gpu_dlls()
        providers = [(p[0], {**p[1], "trt_cuda_graph_enable": True}) if isinstance(p, tuple) else p
                     for p in providers]
        self.buckets = sorted(buckets)
        self.slots = {}
        for b in self.buckets:
            sess = rt.InferenceSession(file_path, providers=providers)
            self.input = sess.get_inputs()[0]
            self.outputs = sess.get_outputs()
            host = np.zeros((b, *self.input.shape[1:]), np.float32)
            x = rt.OrtValue.ortvalue_from_numpy(host, "cuda", 0)
            outs = [rt.OrtValue.ortvalue_from_shape_and_type([b, *o.shape[1:]], np.float32, "cuda", 0)
                    for o in self.outputs]
            io = sess.io_binding()
            io.bind_ortvalue_input(self.input.name, x)
            for o, v in zip(self.outputs, outs):
                io.bind_ortvalue_output(o.name, v)
            options = rt.RunOptions()
            self.slots[b] = (sess, host, x, outs, io, options)
            sess.run_with_iobinding(io, options)
            sess.run_with_iobinding(io, options)

    def bucket_for(self, n):
        for b in self.buckets:
            if n <= b:
                return b
        raise ValueError(f"batch {n} above the largest bucket {self.buckets[-1]}")

    def run(self, output_names, input_feed, run_options=None):
        rows = input_feed[self.input.name]
        n = len(rows)
        sess, host, x, outs, io, options = self.slots[self.bucket_for(n)]
        host[:n] = rows
        host[n:] = 0
        x.update_inplace(host)
        sess.run_with_iobinding(io, options)
        by_name = {o.name: v for o, v in zip(self.outputs, outs)}
        return [by_name[name].numpy()[:n] for name in output_names]


def timeit(fn, reps=300):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return 1e6 * (time.perf_counter() - t0) / reps


if __name__ == "__main__":
    from Client_Server import make_engine
    model_dir, pmax = sys.argv[1], int(sys.argv[2])
    buckets = [int(b) for b in sys.argv[3].split(",")] if len(sys.argv) > 3 else [8, 16, 32, 48, 64, 96]
    shape = Gomoku().get_input_state().shape
    plain = make_engine(model_path(model_dir, pmax), providers_for(model_dir, shape, pmax), pmax)
    graph = GraphEngine(model_path(model_dir, pmax), providers_for(model_dir, shape, pmax), buckets)
    names = [o.name for o in plain.sess.get_outputs()]
    rng = np.random.default_rng(0)
    worst = 0.0
    sizes = [1, 3, 8, 9, 16, 17, 31, 39, 48, 50, 64, 80, 96, 5, 96, 1, 40, 8]
    for b in sizes:
        if b > max(buckets):
            continue
        for _ in range(3):
            x = rng.integers(0, 2, (b, *shape)).astype(np.float32)
            p = plain.run(names, {"inputs": x})
            g = graph.run(names, {"inputs": x})
            assert all(a.shape == c.shape for a, c in zip(p, g)), (b, [a.shape for a in p], [c.shape for c in g])
            worst = max(worst, max(float(np.abs(a - c).max()) for a, c in zip(p, g)))
    print(f"correctness over sizes {sizes}: max|graph - plain| = {worst:.3g}")
    for b in buckets:
        x = rng.integers(0, 2, (b, *shape)).astype(np.float32)
        p = plain.run(names, {"inputs": x})[0]
        g = graph.run(names, {"inputs": x})[0]
        bad = np.where(np.abs(p - g).max(axis=1) > 1e-2)[0]
        stale = graph.run(names, {"inputs": rng.integers(0, 2, (b, *shape)).astype(np.float32)})[0]
        print(f"bucket {b:3d}: wrong rows {len(bad)}/{b} (first {bad[:3].tolist()}), "
              f"output changes with new input: {float(np.abs(stale - g).max()) > 1e-3}")
    print(f"{'batch':>5} {'plain us':>9} {'graph us':>9} {'speedup':>7}")
    for b in (1, 8, 16, 32, 39, 48, 64, 96):
        if b > max(buckets):
            continue
        x = rng.integers(0, 2, (b, *shape)).astype(np.float32)
        tp = timeit(lambda: plain.run(names, {"inputs": x}))
        tg = timeit(lambda: graph.run(names, {"inputs": x}))
        print(f"{b:5d} {tp:9.0f} {tg:9.0f} {tp / tg:6.2f}x")
