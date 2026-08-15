"""Stage 2 verification: hand-built trees through the REAL MCTS with a mock net
whose values are controlled, checking every sign against hand computation.

Convention under test: net value = expected outcome for the player ABOUT TO MOVE
at the evaluated state. Input plane 0 = player who JUST moved (so about-to-move
= -plane0).
"""
import sys, os
sys.path.insert(0, r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero")
import numpy as np

from MCTS import MCTS
from Connect4.game import Connect4


class MockSession:
    """Mimics rt.InferenceSession.run. value_fn maps the (6,7,4) input -> scalar."""
    def __init__(self, value_fn):
        self.value_fn = value_fn
        self.calls = 0

    def run(self, output_names, input_feed):
        self.calls += 1
        x = input_feed["inputs"][0]          # (6,7,4) float32
        v = float(self.value_fn(x))
        pol = (np.ones(7, dtype=np.float32) / 7.0)[None]
        return [pol, np.array([[v]], dtype=np.float32)]


def new_mcts(game, sess):
    return MCTS(game, session=sess, use_njit=False, use_dirichlet=False, tau=0.0,
                fast_find_win=False)


results = []

# ------------------------------------------------------------------ Test A1
# Constant net value 0.7 on an empty board (mover = -1). Exactly 7 sims expand
# each root child once. At each child the player about to move is +1, so the net
# claims +1 wins 0.7 => root child Q (from mover -1's view) must be -0.7 for all.
game = Connect4()
sess = MockSession(lambda x: 0.7)
m = new_mcts(game, sess)
m.run(iteration_limit=7, use_bar=False)
cv = np.array(m.root.child_values, dtype=np.float64)
cn = np.array(m.root.child_visits, dtype=np.float64)
ok = np.allclose(cn, 1.0) and np.allclose(cv, -0.7)
results.append(("A1 const-value first-backup sign", ok,
                f"child_visits={cn.tolist()} child_values={cv.tolist()} expected all -0.7"))

# ------------------------------------------------------------------ Test A2
# Consistent oracle: net always claims player -1 wins 0.9, expressed in the
# to-move perspective: value = +0.9 when -1 is to move, -0.9 when +1 is to move.
# to-move = -plane0. If every sign in select/expand/backprop is right, EVERY
# backup reaching the root (mover -1) is worth +0.9, so child_values ==
# 0.9*child_visits EXACTLY at every visited node, at every depth, both parities.
def oracle(x):
    just_moved = float(x[0, 0, 0])   # plane 0 constant
    to_move = -just_moved
    return 0.9 * (1.0 if to_move == -1.0 else -1.0)

game = Connect4()
sess = MockSession(oracle)
m = new_mcts(game, sess)
m.run(iteration_limit=400, use_bar=False)

worst = [0.0]
def walk(node, depth, sign):
    # sign: +1 if the chooser at `node` is player -1 (root mover), else -1
    if not hasattr(node, "children") or not node.children:
        return
    cv = np.array(node.child_values, dtype=np.float64)
    cn = np.array(node.child_visits, dtype=np.float64)
    mask = cn > 0
    if mask.any():
        # the mock's 0.9 is truncated to float32 by the session contract; compare
        # against that exact value. A sign flip anywhere would deviate by >= 1.8
        # per visit, 6+ orders of magnitude above float noise.
        v32 = float(np.float32(0.9))
        dev = np.abs(cv[mask] - sign * v32 * cn[mask]).max()
        worst[0] = max(worst[0], dev)
    for ch in node.children:
        if getattr(ch, "is_terminal", None) is None:
            walk(ch, depth + 1, -sign)

walk(m.root, 0, +1)
ok = worst[0] < 1e-9
results.append(("A2 oracle consistency all depths", ok,
                f"max |child_values - expected| over whole tree = {worst[0]:.2e} "
                f"(root visits={m.root.visits}, net calls={sess.calls})"))

# ------------------------------------------------------------------ Test A3
# Win-in-1 for the mover at the root. -1 has 3 in a row on the bottom row
# (cols 0..2), col 3 empty, -1 to move. Expect: chosen move col 3, recorded
# q (winrate, index 2 of move_probs row) == +1.0 (mover perspective).
game = Connect4()
for a in (0, 6, 1, 6, 2):   # -1: 0,1,2 ; +1: 6,6
    game.do_action(a)
game.do_action(6)           # +1 plays 6 again -> -1 to move with 3 in a row
assert game.get_next_player() == -1
sess = MockSession(lambda x: 0.0)
m = new_mcts(game, sess)
move, probs = m.run(iteration_limit=50, use_bar=False)
top = probs[0]
ok = (move == 3) and abs(top[2] - 1.0) < 1e-12 and top[7] == -1
results.append(("A3 terminal root: q=+1 for winning mover", ok,
                f"move={move} winrate={top[2]} is_terminal={top[7]} (net calls={sess.calls})"))

# ------------------------------------------------------------------ Test A4
# Mover is LOST: +1 has an open-ended 3 (cols 1,2,3 bottom) and it is -1's
# move. -1 cannot win first; whatever -1 plays, +1 has a win-in-1, found by
# _expand_with_terminal_actions one ply down. Every root child Q must be -1.
game = Connect4()
for a in (0, 1, 0, 2):      # -1: 0,0 ; +1: 1,2
    game.do_action(a)
game.do_action(5)           # -1 plays 5
game.do_action(3)           # +1 completes cols 1,2,3 bottom -> open at 0? col0 taken; open at 4
# board bottom: -1 at 0, +1ils at 1,2,3, -1 at 5. +1 threatens col 4 (win). also 0 stacked?
assert game.get_next_player() == -1
sess = MockSession(lambda x: 0.0)
m = new_mcts(game, sess)
move, probs = m.run(iteration_limit=100, use_bar=False)
qs = np.array([row[2] for row in probs], dtype=np.float64)
visits = np.array([row[4] for row in probs])
# the block at col 4 delays: after -1 blocks col4, +1 does not have an immediate
# win, so ONLY the non-blocking children are proven lost (-1 exactly). The
# blocking child gets net values (0.0 here). Check: every child except the
# col-4 block has Q == -1 exactly.
acts = [row[0] for row in probs]
bad = []
for a, q, n in zip(acts, qs, visits):
    if n == 0:
        continue
    if a == 4:
        continue
    if abs(q - (-1.0)) > 1e-12:
        bad.append((int(a), float(q), int(n)))
ok = len(bad) == 0
results.append(("A4 opponent win-in-1 propagates as Q=-1", ok,
                f"non-block children with Q != -1: {bad}; all: " +
                str([(int(a), round(float(q), 3), int(n)) for a, q, n in zip(acts, qs, visits)])))

# ------------------------------------------------------------------ Test A5
# Perspective of the value fed at expansion == to-move player of the state the
# net actually saw. Wire a net that RECORDS plane0 of every state it is asked
# to evaluate; then confirm each call's plane0 equals the player who just moved
# in that node's action_history parity (root at ply p: history length known).
game = Connect4()
game.do_action(3)  # ply 1, +1 to move now
seen = []
def spy(x):
    seen.append((float(x[0, 0, 0]), int(np.abs(x[..., 3]).sum())))  # (plane0, stones)
    return 0.0
sess = MockSession(spy)
m = new_mcts(game, sess)
m.run(iteration_limit=30, use_bar=False)
# stones on evaluated board = ply of that state; player who made the LAST move
# at ply k (k stones) is -1 if k odd, +1 if k even... move order: ply1 by -1,
# ply2 by +1 => last mover at k stones = -1 when k odd, +1 when k even (k>0).
bad = [(p0, k) for p0, k in seen if k > 0 and p0 != (-1.0 if k % 2 == 1 else 1.0)]
ok = len(bad) == 0 and len(seen) > 5
results.append(("A5 plane0 == just-moved player for every net call", ok,
                f"{len(seen)} net calls, mismatches: {bad[:5]}"))

print()
for name, ok, detail in results:
    print(("PASS " if ok else "FAIL ") + name)
    print("      " + detail)
print()
print("ALL PASS" if all(r[1] for r in results) else "SOME FAILED")
