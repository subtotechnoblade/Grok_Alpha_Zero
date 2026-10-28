"""engine.run latency by batch size for one TensorRT profile, plus the cost of an Event.set.
python -m tests.profile_engine <model_dir> <profile_max>"""
import multiprocessing as mp
import sys
import time

import numpy as np

from tests.mcts_scenarios import ROOT
from tests.bench_search_speed import providers_for, model_path
from Gomoku.game import Gomoku


def timeit(fn, reps):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return 1e6 * (time.perf_counter() - t0) / reps


if __name__ == "__main__":
    from Client_Server import make_engine
    model_dir, pmax = sys.argv[1], int(sys.argv[2])
    shape = Gomoku().get_input_state().shape
    providers = providers_for(model_dir, shape, pmax)
    if len(sys.argv) > 3 and sys.argv[3] == "graph":
        providers[0][1]["trt_cuda_graph_enable"] = True
    engine = make_engine(model_path(model_dir, pmax), providers, pmax)
    print("providers:", engine.sess.get_providers())
    names = [o.name for o in engine.sess.get_outputs()]
    for b in sorted({1, 2, 4, 8, 16, 24, 32, 39, 48, 64, 96, pmax}):
        if b > pmax:
            continue
        x = np.random.randint(0, 2, (b, *shape)).astype(np.float32)
        us = timeit(lambda: engine.run(names, {"inputs": x}), 300)
        print(f"batch {b:3d}: {us:7.0f} us  {us / b:6.1f} us/row")
    event = mp.Event()
    print(f"Event.set: {timeit(event.set, 10000):.1f} us, Event.clear: {timeit(event.clear, 10000):.1f} us")
    out = np.empty((8, 225), np.float32)
    src = np.random.rand(39, 225).astype(np.float32)
    print(f"copy 8x225 rows: {timeit(lambda: out.__setitem__(slice(None), src[8:16]), 10000):.1f} us")
