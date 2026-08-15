"""How did the net's opening preference evolve across generations 0..9?

Two views per generation:
  1. raw prior: model.onnx policy + value on the empty board (what the net alone says)
  2. searched:  375-sim MCTS visit distribution at ply 0, play-time settings
                (no dirichlet, tau=0), gen 9 only - what the user actually sees

Column 3 is the winning opening. If the prior drifts toward column 1 over
generations, self play data taught it that; if the prior is fine but the search
picks 1, the problem is at play time.
"""
import os
import sys
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import onnxruntime as rt

rt.set_default_logger_severity(4)

from Connect4.game import Connect4

game = Connect4()
empty = game.get_input_state()[None].astype(np.float32)

print(f"{'gen':>4} {'value':>7}  policy per column 0..6 (raw prior, empty board)")
for d in sorted(glob.glob("Connect4/Grok_Zero_Train/*"),
                key=lambda p: int(os.path.basename(p))):
    m = os.path.join(d, "model.onnx")
    if not os.path.exists(m):
        continue
    s = rt.InferenceSession(m, providers=["CPUExecutionProvider"])
    pol, val = s.run(None, {"inputs": empty})
    cols = " ".join(f"{p:6.3f}" for p in pol[0])
    star = int(np.argmax(pol[0]))
    print(f"{os.path.basename(d):>4} {val[0][0]:>7.3f}  [{cols}]  argmax={star}")

print()
print("=== gen 9, searched: 375 sims, no dirichlet, tau=0 (play.py conditions) ===")
from MCTS import MCTS

sess = rt.InferenceSession("Connect4/Grok_Zero_Train/9/model.onnx",
                           providers=["CPUExecutionProvider"])
for trial in range(3):
    g = Connect4()
    tree = MCTS(game=g, session=sess, use_njit=None,
                c_puct_init=1.5, use_dirichlet=False, tau=0, fast_find_win=False)
    action, probs = tree.run(iteration_limit=375, use_bar=False)
    dist = {int(np.ravel(a)[0]) if np.ndim(a) else int(a): p for a, p, *rest in probs}
    row = " ".join(f"{dist.get(c, 0.0):6.3f}" for c in range(7))
    print(f"  trial {trial}: picks column {int(np.ravel(action)[0]) if np.ndim(action) else int(action)}   visits [{row}]")
