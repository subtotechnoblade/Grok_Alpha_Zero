"""Unit probes for player-symmetry of the no-net MCTS path.

Probe A: mirror-symmetry of the search itself. Take a position P and its
color-swapped mirror -P (same stones, colors flipped, side to move flipped).
Run MCTS(session=None) on both with the same seed. If the code is
player-symmetric, the visit distributions must be identical in distribution.
We test the *forced* cases exactly: immediate win, forced block.

Probe B: sign of the backed-up value one ply above a loss. After a move that
gives the opponent a mate-in-1, the root edge Q must be negative for BOTH
colors.

Probe C: ply-42 win credit. Construct a position with 41 stones where the last
move wins for p+1 -> get_terminal_actions must return mask 1 and is_terminal +1.
"""
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero")
sys.path.insert(0, str(ROOT))

import numpy as np
from Connect4.game import Connect4
from MCTS import MCTS


def game_from_moves(moves):
    g = Connect4()
    for m in moves:
        g.do_action(m)
    return g


def run_probe(moves, sims=375, seed=0, label=""):
    np.random.seed(seed)
    g = game_from_moves(moves)
    m = MCTS(game=g, session=None, use_njit=True, c_puct_init=1.5,
             use_dirichlet=True, dirichlet_alpha=1.33, dirichlet_epsilon=0.25,
             tau=0.0, fast_find_win=False)
    action, probs = m.run(iteration_limit=sims, time_limit=None, use_bar=False)
    # probs rows: [action, prob, winrate, value, visits, prob_prior, root_visits, is_terminal]
    print(f"{label} next_player={g.next_player} chosen={action}")
    for row in probs:
        print(f"   a={row[0]} visits={row[4]:>4} Q={row[2]:+.3f} prior={row[5]:.3f} term={row[7]}")
    return action, probs


print("=== Probe A1: p-1 has mate-in-1 (three in a column at col 0) ===")
# p-1: 0,0,0  p+1: 1,1  -> p-1 to move, col 0 wins
run_probe([0, 1, 0, 1, 0, 2], sims=100, seed=1, label="p-1 mate-in-1")
# mirror: p+1 has the same mate. moves: swap colors by inserting a waiting pair?
# exact color swap isn't reachable by move list; instead give p+1 the identical pattern:
# p-1: 5,5  p+1: 0,0  then p-1: 6 -> p+1 to move with 0,0 stack... need 3 in col 0 for p+1
print("\n=== Probe A2: p+1 has mate-in-1 (three in a column at col 0) ===")
# plies: p-1:1, p+1:0, p-1:2, p+1:0, p-1:1, p+1 to move has 2... need one more pair
run_probe([1, 0, 2, 0, 3, 0, 5], sims=100, seed=1, label="pre p+1 mate")  # after 7 plies p+1 to move? 7 plies -> p+1 to move
# above: p+1 stones at col0 x3 -> mate-in-1 at col 0? col0 has 3 p+1 stones. yes.

print("\n=== Probe B1: p-1 to move, must block p+1's three-in-column at col 0 ===")
_, probs = run_probe([1, 0, 2, 0, 3], sims=375, seed=2, label="p-1 must block")
print("\n=== Probe B2: p+1 to move, must block p-1's three-in-column at col 0 ===")
_, probs = run_probe([0, 1, 0, 2, 0, 3, 6, 6], sims=375, seed=2, label="p+1 must block")
# after moves: p-1 col0 x3? plies0,2,4 = 0,0,0 by p-1; p+1: 1,2,3; then p-1:6, p+1:6 -> 8 plies, p-1 to move
# oops that's p-1 to move again. fix: [0,1,0,2,0] is 5 plies, p+1 to move, p-1 threatens col0.
print("   (fixing B2 with a 5-ply position)")
_, probs = run_probe([0, 1, 0, 2, 0], sims=375, seed=2, label="p+1 must block")

print("\n=== Probe C: full-board last-move win credit (ply 42) ===")
# Build a 41-stone board where col 3 has one slot left and moving there wins for the mover.
# Construct explicitly rather than via legal move list: fill board pattern with no 4-in-row,
# then check get_terminal_actions style call directly.
g = Connect4()
# checkerboard-ish column fill that avoids wins is fiddly; do a direct kernel check instead:
b = np.zeros((6, 7), dtype=np.int8)
# column pattern that avoids vertical 4s: repeating -1,-1,1,1,-1,1 etc. We just need ONE
# concrete legal-ish board; wins before now are irrelevant for the kernel sign check.
# p+1 has three in a row at row 5 cols 0-2, empty at (5,3):
b[5, 0] = b[5, 1] = b[5, 2] = 1
res_p1 = Connect4.check_win_MCTS(
    Connect4.do_action_MCTS(b.copy(), 3, 1), 1, np.array([3], dtype=np.int8))
b2 = -b
res_m1 = Connect4.check_win_MCTS(
    Connect4.do_action_MCTS(b2.copy(), 3, -1), -1, np.array([3], dtype=np.int8))
print(f"   horizontal win credit: p+1 -> {res_p1} (expect 1), p-1 -> {res_m1} (expect -1)")

# draw-vs-win ordering: winning stone that also fills the board must be a WIN not a draw
full = np.array([
    [1, 1, -1, 0, -1, -1, 1],
    [-1, -1, 1, -1, 1, 1, -1],
    [1, 1, -1, 1, -1, -1, 1],
    [-1, -1, 1, -1, 1, 1, -1],
    [1, 1, -1, 1, -1, -1, 1],
    [-1, -1, 1, 1, 1, -1, -1]], dtype=np.int8)
# (3,3) empty; placing p+1 at col 3 -> vertical 4 in col3 rows 2..5? col3 rows5,4,3?,2...
col3 = full[:, 3]
print(f"   col3 before: {col3}")
after = Connect4.do_action_MCTS(full.copy(), 3, 1)
res = Connect4.check_win_MCTS(after, 1, np.array([3], dtype=np.int8))
print(f"   last-slot winning move result: {res} (expect 1, NOT 0-draw)")
