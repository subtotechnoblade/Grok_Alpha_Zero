"""Tests for the persistent-worker rewrite. Headless, no GPU, no model file, ~15s.

Covers the four tests in tmp/REWRITE_PLAN.md section 4:
  1. Cache_Wrapper unit behaviour
  2. generation 0 end to end (session=None path)
  3. generation > 0 end to end through the real shared memory / inference server wiring
  4. resume: a short generation tops itself up on re-run

Everything runs in a scratch directory. The real Grok_Zero_Train folders are never touched.

    python tmp/test_rewrite.py
"""
import os, sys, shutil, tempfile, types, traceback
os.environ["NUMBA_DISABLE_JIT"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import h5py as h5

# Stub onnxruntime before anything imports it, so no model file or GPU is needed.
_fake = types.ModuleType("onnxruntime")


class FakeSession:
    """Deterministic per-input outputs, so a mixed-up response is detectable."""

    def __init__(self, path, providers=None, **kw):
        pass

    def run(self, names, input_feed):
        x = np.asarray(input_feed["inputs"], dtype=np.float32)
        b = x.shape[0]
        s = x.reshape(b, -1).sum(1)
        policy = np.abs(np.sin(s[:, None] + np.arange(7, dtype=np.float32)[None, :])) + 1e-3
        policy = (policy / policy.sum(1, keepdims=True)).astype(np.float32)
        value = np.tanh(s[:, None] * 0.01).astype(np.float32)
        return [policy, value]


_fake.InferenceSession = FakeSession
_fake.SessionOptions = object
_fake.get_available_providers = lambda: ["CPUExecutionProvider"]
sys.modules["onnxruntime"] = _fake

from Session_Cache import Cache_Wrapper
from Self_Play import run_self_play
from Connect4.Connect4 import Connect4, build_config as C4_BUILD, train_config as C4_TRAIN, optimizer_config

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail and not cond else ""))


# ---------------------------------------------------------------- 1. memo unit test
def test_cache():
    print("\n1. Cache_Wrapper")
    calls = {"n": 0}

    class Counting(FakeSession):
        def __init__(self):
            pass

        def run(self, names, input_feed):
            calls["n"] += 1
            return FakeSession.run(self, names, input_feed)

    c = Cache_Wrapper(Counting(), max_cache_depth=3, max_entries=5)
    x = np.ones((1, 6, 7, 4), dtype=np.float32)

    a = c.run(["policy", "value"], {"inputs": x}, depth=0)
    b = c.run(["policy", "value"], {"inputs": x.copy()}, depth=0)
    check("same input at shallow depth is a hit", c.hits == 1 and c.misses == 1 and calls["n"] == 1)
    check("hit returns identical values", np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))

    before = (c.hits, c.misses, calls["n"])
    c.run(["policy", "value"], {"inputs": x}, depth=3)
    c.run(["policy", "value"], {"inputs": x}, depth=99)
    check("depth >= max_cache_depth bypasses the memo entirely",
          (c.hits, c.misses) == before[:2] and calls["n"] == before[2] + 2)

    for i in range(20):
        c.run(["policy", "value"], {"inputs": np.full((1, 6, 7, 4), i, dtype=np.float32)}, depth=0)
    check("max_entries respected", len(c.cache) == 5, f"got {len(c.cache)}")

    # a caller doing what get_legal_actions_policy_MCTS does must not corrupt the memo
    p, _ = c.run(["policy", "value"], {"inputs": x}, depth=0)
    original = p.copy()
    legal = np.array([0, 2, 4])
    sub = p[0][legal]
    sub /= sub.sum()
    p2, _ = c.run(["policy", "value"], {"inputs": x}, depth=0)
    check("fancy-index + in-place normalise does not mutate cached arrays",
          np.array_equal(p2, original))


# ------------------------------------------------------- shared harness for 2, 3, 4
def make_scratch(tmp, generation):
    folder = os.path.join(tmp, str(generation))
    os.makedirs(folder, exist_ok=True)
    with h5.File(f"{folder}/Self_Play_Data.h5", "w", libver="latest") as f:
        f.create_dataset("game_stats", maxshape=(6,), dtype=np.uint32, data=np.zeros(6,))
    return folder


def cfg(games, workers, cache_depth=10):
    t = dict(C4_TRAIN)
    t.update(games_per_generation=games, num_workers=workers, use_gpu=False,
             use_tensorrt=False, MCTS_iteration_limit=24, max_cache_depth=cache_depth)
    return t


def stats(folder):
    with h5.File(f"{folder}/Self_Play_Data.h5", "r") as f:
        return f["game_stats"][:]


def run(folder, train_cfg, generation):
    run_self_play(Connect4, (C4_BUILD, train_cfg, optimizer_config), folder)
    return stats(folder)


# ------------------------------------------------------------- 2. generation 0 e2e
def test_gen0(tmp):
    print("\n2. generation 0 end to end (session=None)")
    folder = make_scratch(tmp, 0)
    t = cfg(games=12, workers=3)
    t["use_inference_server"] = False
    s = run(folder, t, 0)
    check("all 12 games written", int(s[2]) == 12, f"got {int(s[2])}")
    check("results recorded", int(s[3]) + int(s[4]) + int(s[5]) == 12)
    n_datasets = 0
    with h5.File(f"{folder}/Self_Play_Data.h5", "r") as f:
        n_datasets = (len(f.keys()) - 1) // 3
    check("2 augmented dataset triples per game", n_datasets == 24, f"got {n_datasets}")


# -------------------------------------------------- 3. generation > 0, real shm path
def test_server(tmp):
    print("\n3. generation > 0 through the real inference server + shared memory")
    folder = make_scratch(tmp, 1)
    open(f"{folder}/model.onnx", "w").close()
    t = cfg(games=9, workers=3, cache_depth=10)
    t["use_inference_server"] = True
    s = run(folder, t, 1)
    check("all 9 games written", int(s[2]) == 9, f"got {int(s[2])}")
    check("no Cache/ folder created (memo is in-memory)",
          not os.path.exists(f"{folder}/Cache"))


# -------------------------------------------------------------- 4. resume tops up
def test_resume(tmp):
    print("\n4. resume tops up a short generation")
    folder = make_scratch(tmp, 2)
    t = cfg(games=6, workers=2)
    t["use_inference_server"] = False
    run(folder, t, 2)
    first = int(stats(folder)[2])

    t2 = dict(t); t2["games_per_generation"] = 10
    run(folder, t2, 2)
    second = int(stats(folder)[2])
    check("first pass wrote 6", first == 6, f"got {first}")
    check("re-run topped up to 10 rather than restarting", second == 10, f"got {second}")


def main():
    tmp = tempfile.mkdtemp(prefix="grokzero_rewrite_")
    print(f"scratch: {tmp}")
    try:
        test_cache()
        test_gen0(tmp)
        test_server(tmp)
        test_resume(tmp)
    except Exception:
        traceback.print_exc()
        FAIL.append("uncaught exception")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed" + (f"  -> {FAIL}" if FAIL else ""))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
