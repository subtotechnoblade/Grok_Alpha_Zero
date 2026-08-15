"""Held-out win-in-1 benchmark, parametrised by run root, so run 3 can be compared
against the archived run 2 at MATCHED generation numbers.

  usage: ab_runs.py <root> [maxgen]

Model N is evaluated on gen N's data -- model N trained only on gens N-4..N-1, so
gen N is held out for it. Live-run safe: h5 opened mode="r", OSError skipped.
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
so = rt.SessionOptions()
so.intra_op_num_threads = 2          # a live run owns the rest of the CPU
so.inter_op_num_threads = 1

ROWS, COLS = 6, 7
run_root = sys.argv[1]
maxgen = int(sys.argv[2]) if len(sys.argv) > 2 else 999


def wins_now(board, p):
    for c in range(COLS):
        r = -1
        for rr in range(ROWS - 1, -1, -1):
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


gens = sorted(int(os.path.basename(os.path.dirname(p)))
              for p in glob.glob(f"{run_root}/*/model.onnx"))
gens = [g for g in gens if g <= maxgen]

print(f"run root: {run_root}")
print(f"{'gen':>4} {'n':>6} {'mean V':>9} {'V>0':>8} {'V>0.5':>8} {'V>0.9':>8} {'MSE':>8}"
      f"   (true value = +1 for all; constant-0 scores MSE 1.000)")
for gen in gens:
    fp = f"{run_root}/{gen}/Self_Play_Data.h5"
    if not os.path.exists(fp):
        continue
    try:
        with h5.File(fp, "r") as f:
            n = int(f["game_stats"][2])
            n_sets = (len(f.keys()) - 1) // 3
            if n <= 0 or n_sets % n:
                continue
            aug = n_sets // n
            wins = []
            for gid in range(0, n, 2):
                brd = f[f"boards_{gid * aug}"][:]
                for ply in range(len(brd)):
                    board = brd[ply][..., -1]
                    mover = 1 if int((board == 1).sum()) <= int((board == -1).sum()) else -1
                    if wins_now(board, mover):
                        wins.append(brd[ply])
    except OSError:
        print(f"{gen:>4}   (h5 busy - mid write, skipped)")
        continue
    if not wins:
        continue
    X = np.asarray(wins, dtype=np.float32)
    try:
        sess = rt.InferenceSession(f"{run_root}/{gen}/model.onnx",
                                   sess_options=so, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"{gen:>4}   (model load failed: {type(e).__name__})")
        continue
    out = []
    for i in range(0, len(X), 512):
        _, v = sess.run(None, {"inputs": X[i:i + 512]})
        out.append(v[:, 0])
    v = np.concatenate(out)
    print(f"{gen:>4} {len(v):>6} {v.mean():>+9.3f} {np.mean(v > 0):>8.1%} "
          f"{np.mean(v > 0.5):>8.1%} {np.mean(v > 0.9):>8.1%} "
          f"{float(np.mean((v - 1) ** 2)):>8.3f}")
