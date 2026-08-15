"""Windows + DirectML feasibility benchmark.

Answers three practical questions before committing to a run on the iGPU:
  1. what does use_njit=None (-> False on Windows) cost vs use_njit=True?
  2. what is DirectML inference latency at the batch sizes self play uses?
  3. how long will one generation actually take?

Run with the tensorflow_updated env:
  C:\\Users\\TryHarder\\anaconda3\\envs\\tensorflow_updated\\python.exe tmp/windows_dml_bench.py
"""
import os
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import onnxruntime as rt

from Connect4.Connect4 import Connect4, build_config, train_config, optimizer_config
from MCTS import MCTS

SIMS = int(train_config["MCTS_iteration_limit"] * 1.5)
SCRATCH = f"{ROOT}/tmp/dml_bench"
os.makedirs(SCRATCH, exist_ok=True)
MODEL = f"{SCRATCH}/model.onnx"

# ---------------------------------------------------------------- export current model
if not os.path.exists(MODEL):
    import tensorflow as tf
    from Connect4.Build_Model import build_model
    from To_Onnx import convert_to_onnx
    g = Connect4()
    m = build_model(g.get_input_state().shape, g.policy_shape,
                    (build_config, train_config, optimizer_config))
    print(f"exporting current architecture ({m.count_params():,} params)")
    convert_to_onnx(m, [tf.TensorSpec((None, *g.get_input_state().shape),
                                      tf.float32, name="inputs")], MODEL)

so = rt.SessionOptions()
so.log_severity_level = 3
print("available providers:", rt.get_available_providers())

# ---------------------------------------------------------------- 1. use_njit
print("=" * 74)
print("1. use_njit True vs False (numba is installed and working in this env)")


def play_moves(use_njit, n_moves=8, sims=SIMS):
    np.random.seed(7)
    g = Connect4()
    t = MCTS(g, None, use_njit=use_njit, c_puct_init=train_config["c_puct_init"],
             use_dirichlet=False, tau=0.0, fast_find_win=False)
    jitted = t.get_terminal_actions is not t.get_terminal_actions_fn
    t0 = time.perf_counter()
    for _ in range(n_moves):
        a, _ = t.run(sims, use_bar=False)
        g.do_action(a)
        if g.check_win() != -2:
            break
        t.prune_tree(a)
    return time.perf_counter() - t0, jitted


for u in (True, False):
    play_moves(u, n_moves=2, sims=60)          # warm / compile
res = {}
for u in (True, False):
    dt, jitted = play_moves(u)
    res[u] = dt
    print(f"   use_njit={str(u):<5}  get_terminal_actions jitted={jitted}   "
          f"{dt:6.2f}s for 8 moves   {8 * SIMS / dt:8,.0f} sims/s")
print(f"   -> config has use_njit=None, which is {hasattr(os,'fork')} on Windows.")
print(f"      cost of leaving it None: {res[False] / res[True]:.2f}x slower")

# ---------------------------------------------------------------- 2. DirectML latency
print("=" * 74)
print("2. DirectML inference latency")
sess = rt.InferenceSession(MODEL, sess_options=so,
                           providers=["DmlExecutionProvider", "CPUExecutionProvider"])
cpu = rt.InferenceSession(MODEL, sess_options=so, providers=["CPUExecutionProvider"])
print(f"   dml session providers: {sess.get_providers()}")


def lat(s, b, n=200):
    x = np.zeros((b, 6, 7, 4), dtype=np.float32)
    for _ in range(60):
        s.run(["policy", "value"], {"inputs": x})
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        s.run(["policy", "value"], {"inputs": x})
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e6


W = train_config["num_workers"]
print(f"   {'batch':>6} {'DML us':>10} {'CPU us':>10} {'DML st/s':>10} {'CPU st/s':>10}")
dml1 = None
for b in (1, 4, W, 16):
    d, c = lat(sess, b), lat(cpu, b)
    if b == 1:
        dml1 = d
    print(f"   {b:>6} {d:>10.1f} {c:>10.1f} {b / d * 1e6:>10,.0f} {b / c * 1e6:>10,.0f}")

# ---------------------------------------------------------------- 3. generation estimate
print("=" * 74)
print("3. generation time estimate")
# from the linux profile: ~220 network calls per move, ~36 moves per game
CALLS_PER_MOVE, MOVES = 220, 36
games = train_config["games_per_generation"]
mcts_per_move = res[True] / 8
for label, infer_us in (("DirectML", dml1), ("CPU only", lat(cpu, 1))):
    per_move = mcts_per_move + CALLS_PER_MOVE * infer_us / 1e6
    per_game = per_move * MOVES
    serial = per_game * games
    print(f"   {label:>9}: {per_move * 1000:7.0f} ms/move  {per_game:6.1f} s/game  "
          f"{serial / 60:7.1f} min serial  ->  {serial / W / 60:6.1f} min "
          f"with {W} workers (perfect scaling)")
print("   NOTE: perfect scaling is optimistic. This CPU has 6 cores / 12 threads and")
print("   self play was CPU-bound at 95% with 10 workers on the linux box.")
