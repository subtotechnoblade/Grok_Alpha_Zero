"""Establish, from the data alone, which player is to move in a stored state.

ab_heads.py assumes mover = +1 when (board==1).sum() <= (board==-1).sum(),
i.e. +1 moves first. Connect4.game.get_input_state stores
board_state[0] = -next_player, so chan0 is the player who JUST moved and the
player TO MOVE is -chan0 -- which makes -1 the first mover. The two disagree on
every ply. Decide it empirically:

  * the LAST stored state of a decisive game is the position just before the
    winning move, so the player to move there must have a win-in-1 and the
    stored value label must be strongly positive.
  * the stored value label is from the mover's perspective, so its sign has to
    track the eventual winner under the correct convention.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import h5py as h5

ROWS, COLS = 6, 7


def landing(board, c):
    for rr in range(ROWS - 1, -1, -1):
        if board[rr][c] == 0:
            return rr
    return -1


def win_cols(board, p):
    out = []
    for c in range(COLS):
        r = landing(board, c)
        if r < 0:
            continue
        b = board.copy()
        b[r][c] = p
        for dr, dc in ((0, 1), (1, 0), (1, 1), (-1, 1)):
            n = 1
            for s in (1, -1):
                rr, cc = r + dr * s, c + dc * s
                while 0 <= rr < ROWS and 0 <= cc < COLS and b[rr][cc] == p:
                    n += 1
                    rr += dr * s
                    cc += dc * s
            if n >= 4:
                out.append(c)
                break
    return out


fp = sys.argv[1] if len(sys.argv) > 1 else \
    "tmp/run2_archive_gen0-19_oldvaluehead/12/Self_Play_Data.h5"

hits_abheads, hits_chan0, n_last, val_last = 0, 0, 0, []
sign_ok_abheads, sign_ok_chan0, n_sign = 0, 0, 0

with h5.File(fp, "r") as f:
    n = int(f["game_stats"][2])
    aug = ((len(f.keys()) - 1) // 3) // n
    for gid in range(0, min(n, 300)):
        brd = f[f"boards_{gid * aug}"][:]
        val = f[f"values_{gid * aug}"][:].ravel()
        last = brd[-1]
        board = last[..., -1]
        chan0 = int(last[0, 0, 0])

        # convention A (ab_heads.py): +1 moves first
        mover_a = 1 if int((board == 1).sum()) <= int((board == -1).sum()) else -1
        # convention B (game.py): player to move = -chan0
        mover_b = -chan0

        if len(brd) < 6:
            continue
        n_last += 1
        if win_cols(board, mover_a):
            hits_abheads += 1
        if win_cols(board, mover_b):
            hits_chan0 += 1
        val_last.append(val[-1])

        # sign check: value at each ply should be positive for the player who won
        n_sign += 1

print(f"file: {fp}")
print(f"games checked: {n_last}")
print(f"LAST stored state has a win-in-1 for the mover:")
print(f"  convention A  (ab_heads: +1 first)      {hits_abheads / n_last:6.1%}")
print(f"  convention B  (game.py:  -chan0)        {hits_chan0 / n_last:6.1%}")
print(f"mean stored value at last stored ply: {np.mean(val_last):+.3f}  "
      f"(should be near +1 -- the mover is about to win)")

# also: does chan0 alternate with ply, and which sign starts?
with h5.File(fp, "r") as f:
    n = int(f["game_stats"][2])
    aug = ((len(f.keys()) - 1) // 3) // n
    brd = f["boards_0"][:]
    print("\nchan0 by ply (first 8):", [int(brd[p][0, 0, 0]) for p in range(min(8, len(brd)))])
    print("stone counts (+1,-1) by ply (first 8):",
          [(int((brd[p][..., -1] == 1).sum()), int((brd[p][..., -1] == -1).sum()))
           for p in range(min(8, len(brd)))])
