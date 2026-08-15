"""Leakage-free value test: evaluate model N on generation N's data.

Model N *generates* gen N, but is trained only on gens N-4..N-1 -- gen N did not
exist when model N was built. So gen N is a genuine held-out set for model N, and
it is exactly the distribution model N actually plays in. No leakage on any row.

Two ground truths, both certain:
  1. WIN-IN-1: side to move can complete 4-in-a-row now -> true value +1.
  2. OUTCOME SIGN: for decided games, does the net agree with who actually won?
     (|stored target| > 0.25 means the game was decided, since target = 0.8z+0.2q.)
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

ROWS, COLS = 6, 7
BUCKETS = [(0, 9, "0-9"), (10, 21, "10-21"), (22, 41, "22-41")]


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


def collect(gen):
    """(win-in-1 states, all states, their targets, their plies) from gen N."""
    fp = f"Connect4/Grok_Zero_Train/{gen}/Self_Play_Data.h5"
    if not os.path.exists(fp):
        return None
    try:
        with h5.File(fp, "r") as f:
            n = int(f["game_stats"][2])
            n_sets = (len(f.keys()) - 1) // 3
            if n <= 0 or n_sets % n:
                return None
            aug = n_sets // n
            wins, alls, tgts, plies = [], [], [], []
            for gid in range(0, n, 2):
                brd = f[f"boards_{gid * aug}"][:]
                val = f[f"values_{gid * aug}"][:]
                for ply in range(len(brd)):
                    alls.append(brd[ply])
                    tgts.append(float(val[ply][0]))
                    plies.append(ply)
                    board = brd[ply][..., -1]
                    mover = 1 if int((board == 1).sum()) <= int((board == -1).sum()) else -1
                    if wins_now(board, mover):
                        wins.append(brd[ply])
    except OSError:
        return None
    return (np.asarray(wins, dtype=np.float32), np.asarray(alls, dtype=np.float32),
            np.asarray(tgts), np.asarray(plies))


def infer(sess, X):
    out = []
    for i in range(0, len(X), 1024):
        _, v = sess.run(None, {"inputs": X[i:i + 1024]})
        out.append(v[:, 0])
    return np.concatenate(out) if out else np.array([])


GENS = sorted(int(os.path.basename(p)) for p in glob.glob("Connect4/Grok_Zero_Train/*")
              if os.path.exists(os.path.join(p, "model.onnx")))

print("model N evaluated on gen N (held out for it by construction)\n")
print("1. WIN-IN-1 RECOGNITION   true value = +1.0 for every position")
print(f"{'gen':>4} {'n':>6} {'mean V':>9} {'V>0':>8} {'V>0.5':>8} {'V>0.9':>8} {'MSE':>8}")
rows = {}
for gen in GENS:
    got = collect(gen)
    if got is None:
        continue
    wins, alls, tgts, plies = got
    if len(wins) == 0:
        continue
    try:
        sess = rt.InferenceSession(f"Connect4/Grok_Zero_Train/{gen}/model.onnx",
                                   providers=["CPUExecutionProvider"])
    except Exception:
        continue
    v = infer(sess, wins)
    print(f"{gen:>4} {len(v):>6} {v.mean():>+9.3f} {np.mean(v > 0):>8.1%} "
          f"{np.mean(v > 0.5):>8.1%} {np.mean(v > 0.9):>8.1%} "
          f"{float(np.mean((v - 1) ** 2)):>8.3f}")
    rows[gen] = (sess, alls, tgts, plies)

print("\n(a net that always says 0.0 scores MSE 1.000 above)")
print()
print("2. WINNER-SIGN AGREEMENT on decided games, by ply  (50% = coin flip)")
print(f"{'gen':>4} " + " ".join(f"{name:>9}" for _, _, name in BUCKETS) + f"{'overall':>10}")
for gen, (sess, alls, tgts, plies) in rows.items():
    dec = np.abs(tgts) > 0.25
    if dec.sum() == 0:
        continue
    v = infer(sess, alls[dec])
    t, p = tgts[dec], plies[dec]
    cells = []
    for lo, hi, _ in BUCKETS:
        m = (p >= lo) & (p <= hi)
        cells.append(f"{np.mean(np.sign(v[m]) == np.sign(t[m])):>9.1%}" if m.sum()
                     else f"{'-':>9}")
    print(f"{gen:>4} " + " ".join(cells) +
          f"{np.mean(np.sign(v) == np.sign(t)):>10.1%}")
