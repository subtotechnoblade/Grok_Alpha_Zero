"""On-distribution version of the win-in-1 test.

Positions come straight out of the stored self-play data, so "the net was tested
on random garbage it never trains on" cannot explain the result. The stored board
tensor is fed to the net verbatim (it is exactly what self play fed it), and the
side to move is read off the disc counts, so no assumption about canonical
orientation is needed.

Ground truth: if the side to move has a legal column that completes 4-in-a-row,
the true value of that position is +1. No solver or theory required.
"""
import os
import sys
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import h5py as h5
import onnxruntime as rt

rt.set_default_logger_severity(4)

# model N is trained on gens N-4..N-1, so gen 19 data is held out for EVERY
# existing model: model 19 generated it but never trained on it, and models
# 0..18 predate it entirely. Zero leakage for all rows.
SRC_GENS = [int(a) for a in sys.argv[1:]] or [19]
ROWS, COLS = 6, 7


def wins_now(board, p):
    """Does player p have a legal drop that makes 4 in a row?"""
    for c in range(COLS):
        r = -1
        for rr in range(ROWS - 1, -1, -1):      # find the landing row
            if board[rr][c] == 0:
                r = rr
                break
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
                    if n >= 4:
                        return True
                    rr += dr * s
                    cc += dc * s
    return False


states = []
for gen in SRC_GENS:
    fp = f"Connect4/Grok_Zero_Train/{gen}/Self_Play_Data.h5"
    if not os.path.exists(fp):
        continue
    try:
        with h5.File(fp, "r") as f:
            n = int(f["game_stats"][2])
            n_sets = (len(f.keys()) - 1) // 3
            if n <= 0 or n_sets % n:
                continue
            aug = n_sets // n
            for gid in range(0, n, 3):          # subsample games
                brd = f[f"boards_{gid * aug}"][:]
                for ply in range(len(brd)):
                    board = brd[ply][..., -1]
                    pos = int((board == 1).sum())
                    neg = int((board == -1).sum())
                    mover = 1 if pos <= neg else -1
                    if wins_now(board, mover):
                        states.append(brd[ply])
    except OSError:
        continue

X = np.asarray(states, dtype=np.float32)
print(f"on-distribution positions where the SIDE TO MOVE has an immediate win: {len(X)}")
print("true value of every one of these is +1.0\n")
print(f"{'gen':>4} {'mean V':>9} {'V>0':>8} {'V>0.5':>8} {'V>0.9':>8} {'MSE vs +1':>10}")

for gen in sorted(int(os.path.basename(p)) for p in glob.glob("Connect4/Grok_Zero_Train/*")
                  if os.path.exists(os.path.join(p, "model.onnx"))):
    m = f"Connect4/Grok_Zero_Train/{gen}/model.onnx"
    try:
        sess = rt.InferenceSession(m, providers=["CPUExecutionProvider"])
    except Exception:
        continue
    out = []
    for i in range(0, len(X), 1024):
        _, v = sess.run(None, {"inputs": X[i:i + 1024]})
        out.append(v[:, 0])
    v = np.concatenate(out)
    print(f"{gen:>4} {v.mean():>+9.3f} {np.mean(v > 0):>8.1%} {np.mean(v > 0.5):>8.1%} "
          f"{np.mean(v > 0.9):>8.1%} {float(np.mean((v - 1) ** 2)):>10.3f}")

print("\nreference: a net that always outputs 0.0 scores MSE 1.000 here")
