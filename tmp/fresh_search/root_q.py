"""Print the ply-0 root's per-column prior, Q, visits after one 375-iter search (no noise)."""
import sys
import numpy as np
import onnxruntime as rt

sys.path.insert(0, ".")
from MCTS import MCTS
from Connect4.game import Connect4

so = rt.SessionOptions()
so.intra_op_num_threads = 2
so.log_severity_level = 3
sess = rt.InferenceSession("Connect4/Grok_Zero_Train/9/model.onnx",
                           sess_options=so, providers=["CPUExecutionProvider"])
game = Connect4()
m = MCTS(game=game, session=sess, use_njit=None, c_puct_init=1.5,
         use_dirichlet=False, tau=1.0, fast_find_win=False)
m.run(iteration_limit=375, time_limit=None, use_bar=False)
print(f"{'col':>3} {'prior':>7} {'visits':>7} {'Q(mover)':>9}")
rows = []
for c, pp, v, n in zip(m.root.children, m.root.child_prob_priors,
                       m.root.child_values, m.root.child_visits):
    rows.append((int(c.action_history[-1]), pp, n, v / max(n, 1)))
for col, pp, n, q in sorted(rows):
    print(f"{col:>3} {pp:>7.3f} {n:>7d} {q:>9.4f}")
