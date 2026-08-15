"""Two sanity checks the value head has to pass.

(1) PARITY BASELINE. A "net" that ignores the board completely and predicts purely
    from whose turn it is (+ at even plies, - at odd plies, i.e. "the first player
    wins") -- how accurate is that? If the real net can't beat it, the net has
    learned nothing about positions.

(2) MEMORISATION GAP, same model, same task: model N on win-in-1 positions drawn
    from its TRAINING window (gens N-4..N-1) vs from gen N, which it never trained
    on. Any difference is memorisation, not skill.
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


def scan(gen, step=2):
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
            wins, tg, pl = [], [], []
            for gid in range(0, n, step):
                brd = f[f"boards_{gid * aug}"][:]
                val = f[f"values_{gid * aug}"][:]
                for ply in range(len(brd)):
                    tg.append(float(val[ply][0]))
                    pl.append(ply)
                    board = brd[ply][..., -1]
                    mover = 1 if int((board == 1).sum()) <= int((board == -1).sum()) else -1
                    if wins_now(board, mover):
                        wins.append(brd[ply])
    except OSError:
        return None
    return np.asarray(wins, dtype=np.float32), np.asarray(tg), np.asarray(pl)


def infer(sess, X):
    out = []
    for i in range(0, len(X), 1024):
        _, v = sess.run(None, {"inputs": X[i:i + 1024]})
        out.append(v[:, 0])
    return np.concatenate(out) if out else np.array([])


GENS = sorted(int(os.path.basename(p)) for p in glob.glob("Connect4/Grok_Zero_Train/*")
              if os.path.exists(os.path.join(p, "Self_Play_Data.h5")))

print("1. PARITY BASELINE vs THE NET, winner-sign accuracy on decided positions")
print("   baseline = predict '+' at even plies, '-' at odd plies. Uses NO board info.")
print(f"{'gen':>4} {'n decided':>10} {'baseline':>10} {'net':>8} {'net - baseline':>15}")
cache = {}
for gen in GENS:
    got = scan(gen)
    if got is None:
        continue
    wins, tg, pl = got
    cache[gen] = wins
    dec = np.abs(tg) > 0.25
    if dec.sum() == 0:
        continue
    t, p = tg[dec], pl[dec]
    base = np.mean(np.sign(t) == np.where(p % 2 == 0, 1.0, -1.0))
    mp = f"Connect4/Grok_Zero_Train/{gen}/model.onnx"
    if not os.path.exists(mp):
        continue
    # rebuild the decided states for inference
    got2 = scan(gen)
    sess = rt.InferenceSession(mp, providers=["CPUExecutionProvider"])
    # reuse stored boards: re-read cheaply
    with h5.File(f"Connect4/Grok_Zero_Train/{gen}/Self_Play_Data.h5", "r") as f:
        n = int(f["game_stats"][2])
        aug = ((len(f.keys()) - 1) // 3) // n
        X = []
        for gid in range(0, n, 2):
            X.append(f[f"boards_{gid * aug}"][:])
        X = np.concatenate(X).astype(np.float32)
    v = infer(sess, X[dec])
    net = np.mean(np.sign(v) == np.sign(t))
    print(f"{gen:>4} {int(dec.sum()):>10} {base:>10.1%} {net:>8.1%} "
          f"{net - base:>+15.1%}")

print()
print("2. MEMORISATION GAP, same model, win-in-1 MSE (true value +1)")
print(f"{'model':>6} {'train-window MSE':>18} {'held-out MSE':>14} {'gap':>8}")
for gen in GENS[-4:]:
    mp = f"Connect4/Grok_Zero_Train/{gen}/model.onnx"
    if not os.path.exists(mp) or gen not in cache or len(cache[gen]) == 0:
        continue
    sess = rt.InferenceSession(mp, providers=["CPUExecutionProvider"])
    seen = [cache[g] for g in range(gen - 4, gen) if g in cache and len(cache[g])]
    if not seen:
        continue
    Xs = np.concatenate(seen)
    vs, vh = infer(sess, Xs), infer(sess, cache[gen])
    ms = float(np.mean((vs - 1) ** 2))
    mh = float(np.mean((vh - 1) ** 2))
    print(f"{gen:>6} {ms:>18.3f} {mh:>14.3f} {mh - ms:>+8.3f}")
