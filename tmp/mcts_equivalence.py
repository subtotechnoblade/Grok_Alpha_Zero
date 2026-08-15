"""Output equivalence and realistic speedup with a REAL network in the loop.

tmp/mcts_experiment.py runs session=None, where the dummy value is exactly 0.0 and
therefore identical in float32 and float64 - so it proves nothing about FIX 2's
precision change. This script uses a trained onnx model, where child values are
arbitrary float32 and the float64 accumulation genuinely diverges.

Reports:
  1. move agreement and visit-distribution KL between the two modules
  2. speedup with inference on the critical path (the number that matters for
     self play, unlike the pure-MCTS figure)

Usage (WSL): python tmp/mcts_equivalence.py [n_positions]
"""
import glob
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

N_POS = int(sys.argv[1]) if len(sys.argv) > 1 else 24
SIMS = int(train_config["MCTS_iteration_limit"] * 1.5)
PUCT = train_config["c_puct_init"]

gens = sorted(int(os.path.basename(os.path.dirname(p)))
              for p in glob.glob(f"{ROOT}/Connect4/Grok_Zero_Train/*/model.onnx"))
if gens:
    MODEL = f"{ROOT}/Connect4/Grok_Zero_Train/{gens[-1]}/model.onnx"
    WHICH = f"gen {gens[-1]}"
else:
    # training folder cleared for a rerun. any valid model exercises the same code
    # path, so fall back to the pinned snapshot kept by tmp/quantize_int8.py
    MODEL = f"{ROOT}/tmp/quant_out/connect4_fp32.onnx"
    WHICH = "pinned snapshot (no trained gens present)"
so = rt.SessionOptions()
so.intra_op_num_threads = 1
so.log_severity_level = 3


def new_session():
    return rt.InferenceSession(MODEL, sess_options=so,
                               providers=["CPUExecutionProvider"])


rng = np.random.default_rng(7)
POSITIONS = []
while len(POSITIONS) < N_POS:
    g = Connect4()
    depth = int(rng.integers(0, 22))
    ok = True
    for _ in range(depth):
        legal = [c for c in range(7) if int(np.sum(np.abs(g.board[:, c]))) < 6]
        if not legal:
            ok = False
            break
        g.do_action(int(rng.choice(legal)))
        if g.check_win() != -2:
            ok = False
            break
    if ok and (not g.action_history or g.check_win() == -2):
        POSITIONS.append(list(g.action_history))


def build(mod, acts, sess, seed):
    np.random.seed(seed)
    g = Connect4()
    for a in acts:
        g.do_action(a)
    return g, mod.MCTS(g, sess, use_njit=True, c_puct_init=PUCT,
                       use_dirichlet=False, tau=0.0, fast_find_win=False)


def dist(stats):
    d = np.zeros(7)
    for row in stats:
        d[int(row[0])] = float(row[1])
    return d


print("=" * 78)
print(f"MCTS_original vs MCTS.py | REAL network ({WHICH}, CPU EP) | "
      f"{SIMS} sims | {N_POS} positions")

for mod in (base_mod, fixed_mod):     # warm numba + onnx
    g, t = build(mod, [3, 3], new_session(), 0)
    t.run(60, use_bar=False)

print("=" * 78)
print("1. OUTPUT EQUIVALENCE")
agree, kls, vdiff, tvd = 0, [], [], []
for i, acts in enumerate(POSITIONS):
    out = []
    for mod in (base_mod, fixed_mod):
        g, tree = build(mod, acts, new_session(), 5000 + i)
        move, stats = tree.run(SIMS, use_bar=False)
        root_q = float(np.sum(np.asarray(tree.root.child_values, dtype=np.float64)) /
                       max(1.0, float(sum(tree.root.child_visits))))
        out.append((int(move), dist(stats), root_q))
    (m1, d1, q1), (m2, d2, q2) = out
    agree += int(m1 == m2)
    p, q = d1 + 1e-12, d2 + 1e-12
    kls.append(float(np.sum(p * np.log(p / q))))
    tvd.append(float(0.5 * np.abs(d1 - d2).sum()))
    vdiff.append(abs(q1 - q2))

print(f"  chosen move agreement      : {agree}/{len(POSITIONS)} "
      f"({agree / len(POSITIONS) * 100:.1f}%)")
print(f"  visit-distribution KL      : mean {np.mean(kls):.5f}, max {np.max(kls):.5f}")
print(f"  total variation distance   : mean {np.mean(tvd):.4f}, max {np.max(tvd):.4f}")
print(f"  root value estimate |diff| : mean {np.mean(vdiff):.5f}, max {np.max(vdiff):.5f}")
print("  expected: high but not perfect agreement. FIX 2 accumulates child values")
print("  in float64 rather than float32, so PUCT ties can break differently. This is")
print("  a precision increase; the searches are statistically equivalent, not identical.")

print("=" * 78)
print("2. CORRECTNESS INVARIANTS (must be exact)")
CASES = [("forced win",   [0, 6, 1, 6, 2, 6], 3),
         ("forced block", [0, 6, 1, 5, 2],    3),
         ("win over block", [0, 6, 1, 6, 2, 5, 5], 3)]
for label, acts, want in CASES:
    row = []
    for mod in (base_mod, fixed_mod):
        g = Connect4()
        for a in acts:
            g.do_action(a)
        tree = mod.MCTS(g, new_session(), use_njit=True, c_puct_init=PUCT,
                        use_dirichlet=False, tau=0.0, fast_find_win=True)
        mv, _ = tree.run(400, use_bar=False)
        row.append(int(mv))
    ok = all(m == want for m in row)
    print(f"  {label:>16}: base {row[0]}, fixed {row[1]}, want {want}  "
          f"{'OK' if ok else 'FAIL'}")

print("=" * 78)
print("3. SPEEDUP WITH INFERENCE ON THE CRITICAL PATH")
REPS = 3
res = {}
for rep in range(REPS):
    for name, mod in (("MCTS_original", base_mod), ("MCTS.py (optimized)", fixed_mod)):
        sess = new_session()
        t0 = time.perf_counter()
        n = 0
        for i, acts in enumerate(POSITIONS):
            g, tree = build(mod, acts, sess, 9000 + i)
            tree.run(SIMS, use_bar=False)
            n += SIMS
        res.setdefault(name, []).append((time.perf_counter() - t0) / n * 1e6)

b, f = np.median(res["MCTS_original"]), np.median(res["MCTS.py (optimized)"])
print(f"  {'MCTS.py':>14}: {b:7.2f} us/sim  ({1e6 / b:8,.0f} sims/s)")
print(f"  {'MCTS_fixed.py':>14}: {f:7.2f} us/sim  ({1e6 / f:8,.0f} sims/s)")
print(f"  speedup with a real network in the loop: {b / f:.2f}x")
print(f"\n  note: single-threaded CPU inference here. Real self play batches through")
print(f"  the shared-memory server on GPU, so its inference term differs; treat this")
print(f"  as the shape of the effect, not the exact generation number.")
