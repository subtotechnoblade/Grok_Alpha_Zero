"""Follow-up probes: p-1 must-block mirror, and a correct ply-42 win-credit test."""
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero")
sys.path.insert(0, str(ROOT))

import numpy as np
from Connect4.game import Connect4
from MCTS import MCTS


def run_probe(moves, sims=375, seed=2, label=""):
    np.random.seed(seed)
    g = Connect4()
    for m in moves:
        g.do_action(m)
    m = MCTS(game=g, session=None, use_njit=True, c_puct_init=1.5,
             use_dirichlet=True, dirichlet_alpha=1.33, dirichlet_epsilon=0.25,
             tau=0.0, fast_find_win=False)
    action, probs = m.run(iteration_limit=sims, time_limit=None, use_bar=False)
    print(f"{label} next_player={g.next_player} chosen={action}")
    for row in probs:
        print(f"   a={row[0]} visits={row[4]:>4} Q={row[2]:+.3f} prior={row[5]:.3f} term={row[7]}")


print("=== p-1 to move, must block p+1's vertical three at col 5 ===")
# plies: p-1 1, p+1 5, p-1 2, p+1 5, p-1 6, p+1 5 -> p-1 to move, p+1 threatens col5
run_probe([1, 5, 2, 5, 6, 5], label="p-1 must block")

print("\n=== ply-42 win credit, both colors ===")
# full board except (0,3); row0 cols0-2 = mover; last stone completes horizontal 4.
base = np.array([
    [0, 0, 0, 0, 0, 0, 0],       # row 0 filled below
    [-1, 1, -1, 1, -1, 1, -1],
    [1, -1, 1, -1, 1, -1, 1],
    [1, -1, 1, -1, 1, -1, 1],
    [-1, 1, -1, 1, -1, 1, -1],
    [-1, 1, -1, 1, -1, 1, -1]], dtype=np.int8)
for mover in (1, -1):
    b = base.copy()
    b[0] = np.array([mover, mover, mover, 0, -mover, -mover, mover], dtype=np.int8)
    after = Connect4.do_action_MCTS(b.copy(), 3, mover)
    res = Connect4.check_win_MCTS(after, mover, np.array([3], dtype=np.int8))
    filled = np.all(after != 0)
    print(f"   mover={mover:+d} board_full={filled} result={res} (expect {mover})")
    # non-winning last stone -> draw
    b2 = base.copy()
    b2[0] = np.array([mover, mover, -mover, 0, mover, -mover, mover], dtype=np.int8)
    after2 = Connect4.do_action_MCTS(b2.copy(), 3, mover)
    res2 = Connect4.check_win_MCTS(after2, mover, np.array([3], dtype=np.int8))
    print(f"   mover={mover:+d} non-winning last stone result={res2} (expect 0)")
