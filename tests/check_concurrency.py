"""Rows/s from N TensorRT sessions run concurrently from N threads, by batch size.
python -m tests.check_concurrency <model_dir> <profile_max>"""
import sys
import threading
import time

import numpy as np

from tests.mcts_scenarios import ROOT
from tests.bench_search_speed import providers_for, model_path
from Gomoku.game import Gomoku

SECONDS = 4.0


def rows_per_second(sessions, batch, shape):
    names = [o.name for o in sessions[0].get_outputs()]
    x = np.random.default_rng(0).integers(0, 2, (batch, *shape)).astype(np.float32)
    counts = [0] * len(sessions)
    start = threading.Barrier(len(sessions) + 1)
    end = [0.0]

    def loop(i, sess):
        sess.run(names, {"inputs": x})
        start.wait()
        while time.perf_counter() < end[0]:
            sess.run(names, {"inputs": x})
            counts[i] += batch

    threads = [threading.Thread(target=loop, args=(i, s)) for i, s in enumerate(sessions)]
    for t in threads:
        t.start()
    end[0] = time.perf_counter() + SECONDS + 0.05
    start.wait()
    t0 = time.perf_counter()
    for t in threads:
        t.join()
    return sum(counts) / (time.perf_counter() - t0)


if __name__ == "__main__":
    import onnxruntime as rt
    from Client_Server import preload_gpu_dlls
    preload_gpu_dlls()
    model_dir, pmax = sys.argv[1], int(sys.argv[2])
    shape = Gomoku().get_input_state().shape
    sessions = [rt.InferenceSession(model_path(model_dir, pmax), providers=providers_for(model_dir, shape, pmax))
                for _ in range(4)]
    print(f"{'batch':>5} " + " ".join(f"{n} sess rows/s" for n in (1, 2, 3, 4)))
    for batch in (8, 16, 24, 32, 48, 64, 96):
        if batch > pmax:
            continue
        rates = [rows_per_second(sessions[:n], batch, shape) for n in (1, 2, 3, 4)]
        print(f"{batch:5d} " + " ".join(f"{r:13,.0f}" for r in rates) + f"   best {max(rates) / rates[0]:.2f}x")
