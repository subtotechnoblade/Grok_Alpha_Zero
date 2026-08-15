"""Does CPU inference survive a saturated CPU? DirectML vs CPU EP, idle vs loaded.

The earlier DirectML-vs-CPU comparison was run on an IDLE machine, which is not the
deployment condition: self play runs num_workers MCTS processes that saturate every
core, and with use_gpu=False the inference server competes with them for the same
cores. DirectML runs on the iGPU, a separate compute unit, so it should degrade far
less even though it loses on an idle box.

Load is REAL MCTS (session=None), i.e. exactly what a worker does minus inference.

Run:
  C:\\Users\\TryHarder\\anaconda3\\envs\\tensorflow_updated\\python.exe tmp/inference_under_load.py
"""
import multiprocessing as mp
import os
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODEL = f"{ROOT}/tmp/dml_bench/model.onnx"
WARMUP, ITERS = 250, 250


def load_worker(stop):
    """A stand-in self-play worker: real MCTS search, no inference."""
    os.chdir(ROOT)
    import numpy as np
    from Connect4.Connect4 import Connect4
    from MCTS import MCTS
    np.random.seed(os.getpid())
    while not stop.is_set():
        g = Connect4()
        t = MCTS(g, None, use_njit=False, c_puct_init=1.5, use_dirichlet=True,
                 dirichlet_alpha=1.33, tau=1.0, fast_find_win=False)
        for _ in range(42):
            if stop.is_set():
                return
            a, _ = t.run(375, use_bar=False)
            g.do_action(a)
            if g.check_win() != -2:
                break
            t.prune_tree(a)


def bench(provider, batch, fixed_dim):
    import numpy as np
    import onnx
    import onnxruntime as rt
    so = rt.SessionOptions()
    so.log_severity_level = 3
    if provider == "dml":
        so.enable_mem_pattern = False
        so.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        if fixed_dim:
            m = onnx.load(MODEL)
            d0 = m.graph.input[0].type.tensor_type.shape.dim[0]
            if d0.dim_param:
                so.add_free_dimension_override_by_name(d0.dim_param, batch)
        provs = ["DmlExecutionProvider", "CPUExecutionProvider"]
    else:
        provs = ["CPUExecutionProvider"]
    s = rt.InferenceSession(MODEL, sess_options=so, providers=provs)
    x = np.zeros((batch, 6, 7, 4), dtype=np.float32)
    for _ in range(WARMUP):
        s.run(["policy", "value"], {"inputs": x})
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        s.run(["policy", "value"], {"inputs": x})
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e6


def child(q, provider, batch, fixed_dim):
    try:
        q.put(bench(provider, batch, fixed_dim))
    except Exception as e:
        q.put(f"ERR {type(e).__name__}")


def measure(provider, batch, fixed_dim):
    """Isolated process so a DML crash does not kill the run."""
    q = mp.Queue()
    p = mp.Process(target=child, args=(q, provider, batch, fixed_dim))
    p.start()
    try:
        r = q.get(timeout=900)
    except Exception:
        r = "TIMEOUT"
    p.join(30)
    if p.is_alive():
        p.terminate()
    return r


def main():
    N_LOAD = 6                     # Self_Play clamps num_workers to cpu_count()//2 = 6
    CASES = [("cpu", 1, False), ("cpu", 6, False),
             ("dml", 1, True), ("dml", 6, True)]

    print(f"cores: {mp.cpu_count()} logical | load = {N_LOAD} real MCTS processes")
    print(f"{'case':>16} {'idle us':>10} {'loaded us':>11} {'degradation':>13} "
          f"{'loaded st/s':>12}")

    idle = {}
    for prov, b, fd in CASES:
        idle[(prov, b)] = measure(prov, b, fd)

    stop = mp.Event()
    procs = [mp.Process(target=load_worker, args=(stop,)) for _ in range(N_LOAD)]
    for p in procs:
        p.start()
    time.sleep(20)                 # let the load ramp: spawn + numba + first searches

    loaded = {}
    for prov, b, fd in CASES:
        loaded[(prov, b)] = measure(prov, b, fd)

    stop.set()
    for p in procs:
        p.join(20)
        if p.is_alive():
            p.terminate()

    for prov, b, _ in CASES:
        i, l = idle[(prov, b)], loaded[(prov, b)]
        if not isinstance(i, float) or not isinstance(l, float):
            print(f"{prov + ' batch ' + str(b):>16} {str(i):>10} {str(l):>11}")
            continue
        print(f"{prov + ' batch ' + str(b):>16} {i:>10.1f} {l:>11.1f} "
              f"{l / i:>12.2f}x {b / l * 1e6:>12,.0f}")

    print()
    for b in (1, 6):
        c, d = loaded[("cpu", b)], loaded[("dml", b)]
        if isinstance(c, float) and isinstance(d, float):
            win = "DirectML" if d < c else "CPU"
            print(f"  under load, batch {b}: winner = {win} "
                  f"({min(c, d):.1f} us vs {max(c, d):.1f} us, {max(c,d)/min(c,d):.2f}x)")


if __name__ == "__main__":
    main()
