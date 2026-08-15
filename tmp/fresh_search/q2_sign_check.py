"""Q2: verify inherited child stats keep the correct perspective across TWO+ re-rootings.

Method: advance one MCTS (A) through a scripted opening with prune_tree (tree reuse),
then build a fresh MCTS (B) on the identical position. Run both with a big budget,
no dirichlet, tau=0. If the re-rooting handed stats over with a flipped sign, A's
root Q values would anti-correlate with B's and A would pick anti-optimal moves.
Both should agree (up to search noise): positive correlation, same best move.
"""
import sys
import numpy as np
import onnxruntime as rt

sys.path.insert(0, ".")
from MCTS import MCTS
from Connect4.game import Connect4
from Session_Cache import Cache_Wrapper

so = rt.SessionOptions()
so.intra_op_num_threads = 2
so.log_severity_level = 3
raw = rt.InferenceSession("Connect4/Grok_Zero_Train/9/model.onnx",
                          sess_options=so, providers=["CPUExecutionProvider"])
session = Cache_Wrapper(raw, max_cache_depth=100, max_entries=2_000_000)

SCRIPT = [3, 2, 3, 4]   # four re-rootings
BUDGET = 2000

def root_q(mcts):
    out = {}
    for c, v, n in zip(mcts.root.children, mcts.root.child_values, mcts.root.child_visits):
        if n > 0:
            out[int(c.action_history[-1])] = (v / n, n)
    return out

np.random.seed(0)
gameA = Connect4()
A = MCTS(game=gameA, session=session, use_njit=None, c_puct_init=1.5,
         use_dirichlet=False, tau=0.0, fast_find_win=False)
for mv in SCRIPT:
    A.run(iteration_limit=375, time_limit=None, use_bar=False)  # populate children
    gameA.do_action(mv)
    assert gameA.check_win() == -2
    A.prune_tree(mv, False)   # reuse path -> _set_root

print(f"after prune, A root inherited visits = {A.root.visits}, "
      f"child visits = {list(A.root.child_visits)}")
A.run(iteration_limit=BUDGET, time_limit=None, use_bar=False)

gameB = Connect4()
for mv in SCRIPT:
    gameB.do_action(mv)
B = MCTS(game=gameB, session=session, use_njit=None, c_puct_init=1.5,
         use_dirichlet=False, tau=0.0, fast_find_win=False)
B.run(iteration_limit=BUDGET + A.root.visits, time_limit=None, use_bar=False)

qa, qb = root_q(A), root_q(B)
print(f"{'col':>3} {'Q_A(reused)':>12} {'N_A':>6} {'Q_B(fresh)':>12} {'N_B':>6}")
common = sorted(set(qa) & set(qb))
va, vb = [], []
for a in common:
    print(f"{a:>3} {qa[a][0]:>12.4f} {qa[a][1]:>6} {qb[a][0]:>12.4f} {qb[a][1]:>6}")
    va.append(qa[a][0]); vb.append(qb[a][0])
corr = np.corrcoef(va, vb)[0, 1]
bestA = max(qa, key=lambda k: qa[k][1])
bestB = max(qb, key=lambda k: qb[k][1])
print(f"Q correlation reused-vs-fresh: {corr:+.3f}")
print(f"most-visited move: reused={bestA}, fresh={bestB}")
print("PASS" if corr > 0.5 and bestA == bestB else "CHECK MANUALLY")
