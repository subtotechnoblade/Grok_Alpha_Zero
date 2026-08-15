"""MCTS_original.py (pre-optimization) vs MCTS.py with GPU inference on the critical path.

MCTS evaluates one leaf at a time, so this measures BATCH-1 latency across an eGPU
link (PCIe gen3 x4 over Thunderbolt) - the worst case for a discrete GPU and the
honest setting for a single-process search.

Backends: none (Amdahl reference), CUDA fp32, TRT fp32, TRT fp16.
Variants are interleaved within each repeat so GPU clock drift cannot bias one.

Usage (WSL): python tmp/mcts_gpu_experiment.py [reps] [positions]
"""
import os
import sys
import time

import numpy as np
import onnxruntime as rt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from Connect4.Connect4 import Connect4, train_config
sys.path.insert(0, os.path.join(ROOT, "tmp"))
import MCTS_original as base_mod   # pristine pre-optimization copy
import MCTS as fixed_mod            # the live, optimized MCTS.py

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
N_POS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SIMS = int(train_config["MCTS_iteration_limit"] * 1.5)
PUCT = train_config["c_puct_init"]
MODEL = f"{ROOT}/Connect4/Grok_Zero_Train/9/model.onnx"
so = rt.SessionOptions()
so.log_severity_level = 3


def trt_opts(tag, **extra):
    # SEPARATE cache dir per precision. Sharing one makes TRT report "cache result
    # detected as invalid" and rebuild the fp16 engine mid-measurement, which made
    # fp16 look slower than fp32 on the first attempt.
    cache = f"{ROOT}/tmp/quant_out/trt_cache_mcts_{tag}"
    os.makedirs(cache, exist_ok=True)
    return dict({"trt_engine_cache_enable": True, "trt_engine_cache_path": cache,
                 "trt_timing_cache_enable": True, "trt_timing_cache_path": cache,
                 # MCTS only ever submits batch 1
                 "trt_profile_min_shapes": "inputs:1x6x7x4",
                 "trt_profile_opt_shapes": "inputs:1x6x7x4",
                 "trt_profile_max_shapes": "inputs:8x6x7x4"}, **extra)

INT8 = f"{ROOT}/tmp/quant_out/connect4_int8_trt.onnx"   # built by tmp/fix_trt_int8.py

BACKENDS = [
    ("none (Amdahl ref)", None, None),
    ("CUDA fp32", MODEL, [("CUDAExecutionProvider", {}), "CPUExecutionProvider"]),
    ("TRT fp32", MODEL, [("TensorrtExecutionProvider", trt_opts("fp32")),
                         ("CUDAExecutionProvider", {}), "CPUExecutionProvider"]),
    ("TRT fp16", MODEL, [("TensorrtExecutionProvider", trt_opts("fp16", trt_fp16_enable=True)),
                         ("CUDAExecutionProvider", {}), "CPUExecutionProvider"]),
    ("TRT int8", INT8, [("TensorrtExecutionProvider",
                         trt_opts("int8", trt_int8_enable=True, trt_fp16_enable=True)),
                        ("CUDAExecutionProvider", {}), "CPUExecutionProvider"]),
]

# positions used for the search-agreement check (section 2)
AGREE_SEED = 8080

rng = np.random.default_rng(3)
POSITIONS = []
while len(POSITIONS) < N_POS:
    g = Connect4()
    for _ in range(int(rng.integers(0, 16))):
        legal = [c for c in range(7) if int(np.sum(np.abs(g.board[:, c]))) < 6]
        if not legal:
            break
        g.do_action(int(rng.choice(legal)))
        if g.check_win() != -2:
            break
    if not g.action_history or g.check_win() == -2:
        POSITIONS.append(list(g.action_history))


def run_one(mod, sess, acts, seed):
    np.random.seed(seed)
    g = Connect4()
    for a in acts:
        g.do_action(a)
    tree = mod.MCTS(g, sess, use_njit=True, c_puct_init=PUCT, use_dirichlet=False,
                    tau=0.0, fast_find_win=False)
    t0 = time.perf_counter()
    tree.run(SIMS, use_bar=False)
    return time.perf_counter() - t0


def batch1_latency(sess, n=400):
    if sess is None:
        return 0.0
    x = np.zeros((1, 6, 7, 4), dtype=np.float32)
    for _ in range(50):
        sess.run(["policy", "value"], {"inputs": x})
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(["policy", "value"], {"inputs": x})
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e6


print("=" * 84)
print(f"MCTS with GPU inference | connect4 gen 9 | {SIMS} sims | {N_POS} positions "
      f"| {REPS} interleaved reps")
print("batch-1 inference, eGPU over PCIe gen3 x4")
print("=" * 84)
print(f"{'backend':>20} {'infer us':>9} {'base us/sim':>12} {'fixed us/sim':>13} "
      f"{'speedup':>9} {'95% CI':>14} {'infer share':>12}")

results = {}
searches = {}
for label, model_path, providers in BACKENDS:
    try:
        sess = None if providers is None else rt.InferenceSession(
            model_path, sess_options=so, providers=providers)
        if sess is not None:
            got = sess.get_providers()[0]
            sess.run(["policy", "value"], {"inputs": np.zeros((1, 6, 7, 4), np.float32)})
    except Exception as e:
        print(f"{label:>20}  FAILED: {type(e).__name__}: {str(e)[:60]}")
        continue

    # heavy warmup: the TRT engine must be fully built and the GPU clocked up
    # BEFORE any timing, or the first repeat absorbs the build cost
    for mod in (base_mod, fixed_mod):
        for _ in range(3):
            run_one(mod, sess, [3, 3], 1)
    infer_us = batch1_latency(sess)

    per = {"base": [], "fixed": []}
    for r in range(REPS):
        for key, mod in (("base", base_mod), ("fixed", fixed_mod)):
            tot = sum(run_one(mod, sess, a, 400 + i) for i, a in enumerate(POSITIONS))
            per[key].append(tot / (len(POSITIONS) * SIMS) * 1e6)
    b, f = np.array(per["base"]), np.array(per["fixed"])
    ratios = b / f
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    # ~0.7 expansions (hence inference calls) per simulation
    share = infer_us * 0.7 / np.median(f) * 100 if infer_us else 0.0
    results[label] = (infer_us, np.median(b), np.median(f), np.median(ratios))
    print(f"{label:>20} {infer_us:>9.0f} {np.median(b):>12.1f} {np.median(f):>13.1f} "
          f"{np.median(ratios):>8.2f}x  [{lo:.2f},{hi:.2f}] {share:>11.0f}%")

    # what does the SEARCH decide with this backend? (deterministic: tau=0, no noise)
    if sess is not None:
        picks = []
        for i, acts in enumerate(POSITIONS):
            np.random.seed(AGREE_SEED + i)
            g = Connect4()
            for a in acts:
                g.do_action(a)
            tree = fixed_mod.MCTS(g, sess, use_njit=True, c_puct_init=PUCT,
                                  use_dirichlet=False, tau=0.0, fast_find_win=False)
            mv, stats = tree.run(SIMS, use_bar=False)
            d = np.zeros(7)
            for row in stats:
                d[int(row[0])] = float(row[1])
            picks.append((int(mv), d))
        searches[label] = picks
    del sess

print("=" * 84)
print("SEARCH AGREEMENT vs TRT fp32 (does reduced precision change what MCTS plays?)")
ref = searches.get("TRT fp32")
if ref:
    print(f"{'backend':>20} {'same move':>11} {'visit KL':>10} {'visit TVD':>10}")
    for label, picks in searches.items():
        if label == "TRT fp32":
            continue
        same = sum(int(a[0] == b[0]) for a, b in zip(picks, ref))
        kl, tvd = [], []
        for (_, d), (_, dr) in zip(picks, ref):
            p, q = dr + 1e-12, d + 1e-12
            kl.append(float(np.sum(p * np.log(p / q))))
            tvd.append(float(0.5 * np.abs(d - dr).sum()))
        print(f"{label:>20} {same:>6}/{len(ref):<4} {np.mean(kl):>10.4f} "
              f"{np.mean(tvd):>10.4f}")
    print("  a policy that is wrong at the leaf can still be corrected by search;")
    print("  this measures what actually reaches the training target.")

print("=" * 84)
if "TRT fp16" in results and "CUDA fp32" in results:
    i16, _, f16, _ = results["TRT fp16"]
    i32, _, f32, _ = results["CUDA fp32"]
    print(f"fp16 vs CUDA fp32, MCTS_fixed end to end: {f32 / f16:.2f}x "
          f"({f32:.0f} -> {f16:.0f} us/sim); inference alone {i32:.0f} -> {i16:.0f} us")
print("\nCAVEAT: this is ONE process doing batch-1 inference. Real self play runs")
print("num_workers processes through the shared-memory server, which batches their")
print("requests, so the GPU cost is amortised across workers and the inference share")
print("per worker is lower than shown here. Treat these as the pessimistic bound on")
print("the MCTS speedup, and a real generation timing as the truth.")
