"""Profile ONE complete self-play game, start to finish, through Self_Play.play().

Unlike tmp/mcts_phase_profile.py (which isolates the search), this measures the whole
game loop as self play actually runs it: state building, search, policy improvement,
q lookup, do_action, win check, tree pruning, then the post-game augmentation and the
locked h5 write.

Builds the CURRENT architecture and exports it to onnx into a scratch folder, so the
inference cost reflects the model you are about to train, not an old checkpoint.
Nothing in Connect4/Grok_Zero_Train is touched.

Usage (WSL): python tmp/selfplay_game_profile.py [n_games] [--no-trt]
"""
import os
import sys
import time
from collections import Counter

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import h5py as h5
import multiprocessing as mp

from Connect4.Connect4 import Connect4, build_config, train_config, optimizer_config
from Connect4.Build_Model import build_model

N_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else 1
USE_TRT = "--no-trt" not in sys.argv
SCRATCH = f"{ROOT}/tmp/sp_profile"
os.makedirs(SCRATCH, exist_ok=True)
CONFIGS = (build_config, train_config, optimizer_config)
game0 = Connect4()

# ---------------------------------------------------------------- build + export model
if not os.path.exists(f"{SCRATCH}/model.onnx"):
    import tensorflow as tf
    from To_Onnx import convert_to_onnx
    for d in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(d, True)
        except Exception:
            pass
    m = build_model(game0.get_input_state().shape, game0.policy_shape, CONFIGS)
    print(f"exporting current architecture ({m.count_params():,} params) to onnx")
    convert_to_onnx(m, [tf.TensorSpec((None, *game0.get_input_state().shape),
                                      tf.float32, name="inputs")],
                    f"{SCRATCH}/model.onnx")

# ---------------------------------------------------------------- session
import onnxruntime as rt

so = rt.SessionOptions()
so.log_severity_level = 3
avail = rt.get_available_providers()
if USE_TRT and "TensorrtExecutionProvider" in avail:
    cache = f"{SCRATCH}/TRT_cache"
    os.makedirs(cache, exist_ok=True)
    b = "6x7x4"
    w = train_config["num_workers"]
    opts = {"trt_engine_cache_enable": True, "trt_engine_cache_path": cache,
            "trt_timing_cache_enable": True, "trt_timing_cache_path": cache,
            "trt_builder_optimization_level": 5,
            "trt_profile_min_shapes": f"inputs:1x{b}",
            "trt_profile_opt_shapes": f"inputs:1x{b}",
            "trt_profile_max_shapes": f"inputs:{w}x{b}"}
    if train_config.get("trt_fp16", False):
        opts["trt_fp16_enable"] = True
    providers = [("TensorrtExecutionProvider", opts),
                 ("CUDAExecutionProvider", {}), "CPUExecutionProvider"]
    tag = "TRT fp16" if opts.get("trt_fp16_enable") else "TRT fp32"
else:
    providers = ["CPUExecutionProvider"]
    tag = "CPU"
session = rt.InferenceSession(f"{SCRATCH}/model.onnx", sess_options=so, providers=providers)
for _ in range(60):
    session.run(["policy", "value"], {"inputs": np.zeros((1, 6, 7, 4), np.float32)})

# ---------------------------------------------------------------- fresh h5
H5 = f"{SCRATCH}/Self_Play_Data.h5"
if os.path.exists(H5):
    os.remove(H5)
with h5.File(H5, "w", libver="latest") as f:
    f.create_dataset("game_stats", data=np.zeros(6, dtype=np.int64), maxshape=(6,))

# ---------------------------------------------------------------- instrument
import Self_Play as SP
import MCTS as M

acc, cnt = Counter(), Counter()


def wrap(obj, name, key, static=False):
    orig = getattr(obj, name)

    def w(*a, **kw):
        t0 = time.perf_counter_ns()
        try:
            return orig(*a, **kw)
        finally:
            acc[key] += time.perf_counter_ns() - t0
            cnt[key] += 1
    setattr(obj, name, staticmethod(w) if static else w)
    return orig


saved = [
    (M.MCTS, "run", wrap(M.MCTS, "run", "mcts.run")),
    (M.MCTS, "_compute_outputs", wrap(M.MCTS, "_compute_outputs", "  inference")),
    (M.MCTS, "prune_tree", wrap(M.MCTS, "prune_tree", "prune_tree")),
    (M.MCTS, "create_expand_root", wrap(M.MCTS, "create_expand_root", "create_root")),
]
c4 = [(Connect4, n, wrap(Connect4, n, k, static=True)) for n, k in
      (("get_input_state_MCTS", "get_input_state"),
       ("augment_sample", "augment_sample"))]   # both are @staticmethod
inst = [(Connect4, n, wrap(Connect4, n, k)) for n, k in
        (("do_action", "do_action"), ("check_win", "check_win"),
         ("compute_policy_improvement", "policy_improve"))]

# whole-game and the tail after the move loop
play_orig = SP.Self_Play.play


def timed_play(self):
    t0 = time.perf_counter_ns()
    try:
        return play_orig(self)
    finally:
        acc["TOTAL play()"] += time.perf_counter_ns() - t0
        cnt["TOTAL play()"] += 1


SP.Self_Play.play = timed_play

# the locked h5 write: time h5.File open+close inside play
h5open = h5.File


class TimedH5:
    def __init__(self, *a, **kw):
        self.t0 = time.perf_counter_ns()
        self.f = h5open(*a, **kw)

    def __enter__(self):
        return self.f.__enter__()

    def __exit__(self, *a):
        r = self.f.__exit__(*a)
        acc["h5 write"] += time.perf_counter_ns() - self.t0
        cnt["h5 write"] += 1
        return r


SP.h5.File = TimedH5

# ---------------------------------------------------------------- run
lock = mp.Lock()
print("=" * 78)
print(f"ONE self-play game, full Self_Play.play() | inference: {tag} | "
      f"{int(train_config['MCTS_iteration_limit'] * 1.5)} sims/move")
print("=" * 78)

FORCE_NO_BAR = "--no-bar" in sys.argv
if FORCE_NO_BAR:
    # Self_Play.py:99 passes use_bar=True, so every move builds a tqdm bar and
    # updates it once per simulation. This strips it to measure the cost.
    _run = M.MCTS.run

    def no_bar(self, *a, **kw):
        kw["use_bar"] = False
        return _run(self, *a, **kw)
    M.MCTS.run = no_bar
    acc.clear()
    cnt.clear()

wall0 = time.perf_counter()
for i in range(N_GAMES):
    np.random.seed(500 + i)
    SP.Self_Play(Connect4(), session, build_config, train_config,
                 lock, SCRATCH, 1).play()
wall = time.perf_counter() - wall0

with h5.File(H5, "r") as f:
    stats = f["game_stats"][:]
moves = int(stats[1]) // max(1, int(stats[2]))
total = acc["TOTAL play()"] / 1e9

print(f"{N_GAMES} game(s), {int(stats[1])} positions, ~{moves} moves/game, "
      f"{wall:.2f}s wall ({wall / N_GAMES:.2f}s per game)")
print()
print(f"{'segment':>18} {'calls':>8} {'total s':>9} {'ms/call':>9} {'% of game':>10}")
ORDER = ["TOTAL play()", "mcts.run", "  inference", "create_root", "prune_tree",
         "get_input_state", "policy_improve", "do_action", "check_win",
         "augment_sample", "h5 write"]
for k in ORDER:
    if not cnt[k]:
        continue
    s = acc[k] / 1e9
    print(f"{k:>18} {cnt[k]:>8,} {s:>9.3f} {s / cnt[k] * 1000:>9.3f} "
          f"{s / total * 100:>9.1f}%")

known = sum(acc[k] for k in ("mcts.run", "prune_tree", "get_input_state",
                             "policy_improve", "do_action", "check_win",
                             "augment_sample", "h5 write")) / 1e9
print(f"{'-' * 60}")
print(f"{'accounted':>18} {'':>8} {known:>9.3f} {'':>9} {known / total * 100:>9.1f}%")
print(f"{'unaccounted glue':>18} {'':>8} {total - known:>9.3f} {'':>9} "
      f"{(total - known) / total * 100:>9.1f}%")
print()
print(f"inference is {acc['  inference'] / max(1, acc['mcts.run']) * 100:.1f}% of search, "
      f"{acc['  inference'] / max(1, acc['TOTAL play()']) * 100:.1f}% of the game")
print(f"per-move search cost: {acc['mcts.run'] / max(1, cnt['mcts.run']) / 1e6:.1f} ms")

for obj, name, orig in saved + c4 + inst:
    setattr(obj, name, orig)
SP.Self_Play.play = play_orig
SP.h5.File = h5open
