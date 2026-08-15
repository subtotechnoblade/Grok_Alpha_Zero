"""Compare BOTH heads against unambiguous ground truth, per run, per generation.

Model N is evaluated on gen N's data -- held out for it by construction.

Three ground truths, none needing a solver or any theory:
  A. WIN-IN-1  : mover can complete 4-in-a-row now.
       value  -> true value is +1
       policy -> must put mass on a winning column (argmax must be one)
  B. MUST-BLOCK: mover has NO win, opponent has EXACTLY ONE winning column.
       policy -> that column is forced; any other move loses immediately.

  usage: ab_heads.py <run_root> [maxgen]
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
so.intra_op_num_threads = 2
so.inter_op_num_threads = 1

ROWS, COLS = 6, 7
run_root = sys.argv[1]
maxgen = int(sys.argv[2]) if len(sys.argv) > 2 else 999


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


def legal_mask(board):
    return np.array([landing(board, c) >= 0 for c in range(COLS)])


def collect(gen):
    fp = f"{run_root}/{gen}/Self_Play_Data.h5"
    if not os.path.exists(fp):
        return None
    winX, winCols, winMask, blkX, blkCol, blkMask = [], [], [], [], [], []
    try:
        with h5.File(fp, "r") as f:
            n = int(f["game_stats"][2])
            n_sets = (len(f.keys()) - 1) // 3
            if n <= 0 or n_sets % n:
                return None
            aug = n_sets // n
            for gid in range(0, n, 2):
                brd = f[f"boards_{gid * aug}"][:]
                for ply in range(len(brd)):
                    board = brd[ply][..., -1]
                    mover = 1 if int((board == 1).sum()) <= int((board == -1).sum()) else -1
                    mine = win_cols(board, mover)
                    if mine:
                        winX.append(brd[ply])
                        winCols.append(mine)
                        winMask.append(legal_mask(board))
                        continue
                    theirs = win_cols(board, -mover)
                    if len(theirs) == 1:
                        blkX.append(brd[ply])
                        blkCol.append(theirs[0])
                        blkMask.append(legal_mask(board))
    except OSError:
        return "busy"
    return winX, winCols, winMask, blkX, blkCol, blkMask


def infer(sess, X):
    P, V = [], []
    for i in range(0, len(X), 512):
        p, v = sess.run(None, {"inputs": np.asarray(X[i:i + 512], dtype=np.float32)})
        P.append(p)
        V.append(v[:, 0])
    return np.concatenate(P), np.concatenate(V)


gens = sorted(int(os.path.basename(os.path.dirname(p)))
              for p in glob.glob(f"{run_root}/*/model.onnx"))
gens = [g for g in gens if g <= maxgen]

print(f"run root: {run_root}")
print(f"{'':>4} {'':>16} {'--- VALUE on win-in-1 ---':^26} "
      f"{'--- POLICY on win-in-1 ---':^26} {'-- POLICY must-block --':^24}")
print(f"{'gen':>4} {'n_win':>7} {'n_blk':>8} {'mean V':>9} {'V>0':>7} {'MSE':>8} "
      f"{'P(win col)':>12} {'top1 ok':>9} {'block top1':>12} {'P(block)':>10}")

for gen in gens:
    got = collect(gen)
    if got is None:
        continue
    if got == "busy":
        print(f"{gen:>4}   (h5 mid-write, skipped)")
        continue
    winX, winCols, winMask, blkX, blkCol, blkMask = got
    if not winX:
        continue
    try:
        sess = rt.InferenceSession(f"{run_root}/{gen}/model.onnx",
                                   sess_options=so, providers=["CPUExecutionProvider"])
    except Exception:
        continue

    P, V = infer(sess, winX)
    pw, ptop = [], []
    for i, cols in enumerate(winCols):
        p = P[i] * winMask[i]
        s = p.sum()
        p = p / s if s > 0 else p
        pw.append(p[cols].sum())
        ptop.append(int(np.argmax(p)) in cols)

    if blkX:
        PB, _ = infer(sess, blkX)
        btop, bp = [], []
        for i, c in enumerate(blkCol):
            p = PB[i] * blkMask[i]
            s = p.sum()
            p = p / s if s > 0 else p
            btop.append(int(np.argmax(p)) == c)
            bp.append(p[c])
        btop, bp = np.mean(btop), np.mean(bp)
    else:
        btop = bp = float("nan")

    print(f"{gen:>4} {len(winX):>7} {len(blkX):>8} {V.mean():>+9.3f} "
          f"{np.mean(V > 0):>7.1%} {float(np.mean((V - 1) ** 2)):>8.3f} "
          f"{np.mean(pw):>12.1%} {np.mean(ptop):>9.1%} {btop:>12.1%} {bp:>10.1%}")

print("\nrandom-legal baselines: P(win col) and top1 ~ 1/legal (~15-20%);"
      " block top1 ~ same")
