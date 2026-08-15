"""Stage 3+4 verification on REAL stored data (gen 15, read-only).

For each original game (even dataset ids; odd ids are the mirror):
  1. Reconstruct the move sequence from consecutive board planes (plane 3).
  2. Determine the winner INDEPENDENTLY: replay the final move (argmax of the
     stored final policy, tau=0 in the endgame) and run check_win_MCTS.
     T < 42 plies => the game ended by a win; T == 42 with no win => draw.
  3. Recompute z_t (mover-perspective, alternating) from that winner.
  4. Stored target v_t must equal 0.6*z_t + 0.4*q_t with q_t in [-1, 1].
     Solve q_t = (v_t - 0.6*z_t)/0.4 and check the bound. Also run the same
     with z FLIPPED: if the stored perspective were off by one ply or sign,
     the flipped version would fit better / the true one would violate.
  5. Mirror pair: values identical, boards column-flipped, policies flipped.
"""
import sys
sys.path.insert(0, r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero")
import numpy as np
import h5py as h5
from Connect4.game import Connect4

QW = 0.4  # q_weight from Connect4/config.py
PATH = r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero\Connect4\Grok_Zero_Train\15\Self_Play_Data.h5"

try:
    f = h5.File(PATH, mode="r")
except OSError as e:
    print("Could not open h5 read-only:", e)
    sys.exit(1)

n_triples = (len(f.keys()) - 1) // 3
n_games = n_triples // 2
stats = f["game_stats"][:]
print(f"gen15: {stats[2]} games, draws recorded in stats: {stats[4]}")

games_checked = 0
mirror_bad = 0
q_violations = []      # (game, ply, q) with |q| > 1 + eps under the TRUE z
qflip_ok_frac = []     # fraction of plies whose q stays legal under FLIPPED z
draws = 0
winner_fail = 0
eps = 1e-4

all_q = []
all_z = []
all_v = []
ply_of = []

for g in range(n_games):
    b = f[f"boards_{2*g}"][:]        # (T,6,7,4) int8
    p = f[f"policies_{2*g}"][:]      # (T,7)
    v = f[f"values_{2*g}"][:].ravel()  # (T,)

    bm = f[f"boards_{2*g+1}"][:]
    pm = f[f"policies_{2*g+1}"][:]
    vm = f[f"values_{2*g+1}"][:].ravel()

    # ---- mirror check (stage 4)
    if not (np.array_equal(vm, v) and
            np.array_equal(bm, b[:, :, ::-1, :]) and
            np.allclose(pm, p[:, ::-1])):
        mirror_bad += 1

    T = len(b)
    boards = b[..., 3].astype(np.int8)           # (T,6,7) position before each ply
    plane0 = b[:, 0, 0, 0].astype(np.int8)       # just-moved constant
    movers = -plane0                              # player about to move at each ply

    # sanity: movers alternate starting at -1
    expect = np.array([-1, 1] * ((T + 1) // 2))[:T]
    if not np.array_equal(movers, expect):
        print(f"game {g}: mover parity broke: {movers[:8]}")
        continue

    # ---- winner, independently: replay the last stored ply's argmax move
    last_board = boards[-1].copy()
    last_mover = int(movers[-1])
    a = int(np.argmax(p[-1]))
    col = last_board[:, a]
    filled = int(np.sum(np.abs(col)))
    if filled >= 6:
        winner_fail += 1
        continue
    last_board[5 - filled, a] = last_mover
    hist = np.array([a], dtype=np.int64)
    w = int(Connect4.check_win_MCTS(last_board, last_mover, hist))
    total_stones = int(np.abs(last_board).sum())
    if w == -2:
        if total_stones == 42:
            w = 0
        else:
            winner_fail += 1  # could not confirm a terminal move; skip game
            continue
    if w == 0:
        draws += 1

    # ---- recompute z, solve for q
    if w == 0:
        z = np.zeros(T)
    else:
        z = np.where(movers == w, 1.0, -1.0)

    q = (v - (1 - QW) * z) / QW
    qf = (v - (1 - QW) * (-z)) / QW  # deliberately wrong perspective

    viol = np.where(np.abs(q) > 1 + eps)[0]
    for t in viol:
        q_violations.append((g, int(t), float(q[t])))
    qflip_ok_frac.append(float(np.mean(np.abs(qf) <= 1 + eps)))

    all_q.append(q); all_z.append(z); all_v.append(v)
    ply_of.append(np.arange(T))
    games_checked += 1

f.close()

all_q = np.concatenate(all_q); all_z = np.concatenate(all_z); all_v = np.concatenate(all_v)
ply_of = np.concatenate(ply_of)

print(f"\ngames fully checked: {games_checked}/{n_games}  (winner underivable: {winner_fail})")
print(f"draws found by replay: {draws}")
print(f"mirror-pair mismatches: {mirror_bad}")
print(f"\nTRUE z:  plies with implied |q| > 1: {len(q_violations)} / {len(all_q)}")
if q_violations[:10]:
    print("   first violations:", q_violations[:10])
print(f"FLIPPED z: mean fraction of plies still legal: {np.mean(qflip_ok_frac):.3f}"
      f"  (should be far below 1.0 if stored z really is mover-perspective)")
print(f"\nimplied q: mean={all_q.mean():+.3f} std={all_q.std():.3f} "
      f"corr(q,z)={np.corrcoef(all_q, all_z)[0,1]:.3f}")
print(f"stored v: mean={all_v.mean():+.3f} std={all_v.std():.3f}")
dec = all_z != 0
print(f"decided plies: {dec.sum()} ({100*dec.mean():.1f}%)")
print(f"final-3-ply implied q vs z agreement (sign): "
      f"{np.mean(np.sign(all_q[dec & (ply_of >= 30)]) == np.sign(all_z[dec & (ply_of >= 30)])):.3f}")
