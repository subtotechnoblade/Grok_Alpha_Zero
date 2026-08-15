"""E3: ground-truth value accuracy. No floor, no sigma^2, no distributional
assumption -- positions whose true value is known exactly by the rules of Connect4.

  WIN1  : the mover can complete 4-in-a-row right now            -> true value +1
  LOSS2 : the mover has NO immediate win, and the opponent has >=2 distinct
          immediate winning columns. The mover blocks at most one, so the
          opponent wins on the very next move                    -> true value -1

BUG FIXED vs tmp/ab_heads.py:
  ab_heads infers the side to move as `+1 if (board==+1).count <= (board==-1).count`.
  In this data the FIRST player's pieces are -1 (verified over every generation
  below), so at EVEN ply the two counts are equal and that rule returns +1 -- the
  wrong side -- for every even-ply position. Here the mover is read exactly from
  plane 0, which stores -(mover's piece sign).

Each generation is scored separately, and split by whether the position is
byte-identical to one in the evaluated model's own 4-generation training window.
That separates "memorised" from "generalised" on ground truth.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tmp", "learnability"))
os.chdir(REPO)

import numpy as np
import data as D

RUN = "run2"
ROOT = D.ROOTS[RUN]
ROWS, COLS = 6, 7
DIRS = ((0, 1), (1, 0), (1, 1), (-1, 1))


def landing(board, c):
    for r in range(ROWS - 1, -1, -1):
        if board[r][c] == 0:
            return r
    return -1


def win_cols(board, p):
    out = []
    for c in range(COLS):
        r = landing(board, c)
        if r < 0:
            continue
        for dr, dc in DIRS:
            n = 1
            for s in (1, -1):
                rr, cc = r + dr * s, c + dc * s
                while 0 <= rr < ROWS and 0 <= cc < COLS and board[rr][cc] == p:
                    n += 1
                    rr += dr * s
                    cc += dc * s
            if n >= 4:
                out.append(c)
                break
    return out


def collect(gen, verify=False):
    """-> (X_win, X_loss). Mover read from plane 0 == -(mover piece sign)."""
    games = D._read_gen(ROOT, gen)
    win, loss = [], []
    firstsign = {}
    for g in games:
        B = g["boards"]
        for t in range(len(B)):
            st = B[t]
            board = st[..., -1]
            mover = -int(st[..., 0].ravel()[0])
            if verify:
                # parity check: mover must own floor(ply/2) stones
                nm = int((board == mover).sum())
                no = int((board == -mover).sum())
                firstsign[(nm, no)] = firstsign.get((nm, no), 0) + 1
            mine = win_cols(board, mover)
            if mine:
                win.append(st)
                continue
            theirs = win_cols(board, -mover)
            if len(theirs) >= 2:
                loss.append(st)
    return (np.asarray(win, dtype=np.int8) if win else np.zeros((0, 6, 7, 4), np.int8),
            np.asarray(loss, dtype=np.int8) if loss else np.zeros((0, 6, 7, 4), np.int8),
            firstsign)


def onnx_pred(path):
    import onnxruntime as rt
    rt.set_default_logger_severity(4)
    s = rt.InferenceSession(path, providers=["CPUExecutionProvider"])

    def f(X):
        if len(X) == 0:
            return np.zeros(0)
        o = []
        for i in range(0, len(X), 4096):
            _, v = s.run(None, {"inputs": X[i:i + 4096].astype(np.float32)})
            o.append(v[:, 0])
        return np.concatenate(o).astype(np.float64)
    return f


def train_keys(model_gen):
    """byte-keys of the positions model `model_gen` actually trained on."""
    d = D.get(RUN, model_gen - 1)
    return {x.tobytes() for x in d["train"]["X"]}


def main():
    GENS = list(range(9, 17))
    print("collecting ground-truth positions ...")
    sets = {}
    for g in GENS:
        w, l, fs = collect(g, verify=(g == GENS[0]))
        sets[g] = (w, l)
        print(f"  gen {g}: WIN1 {len(w):>6}   LOSS2 {len(l):>6}")
    # mover-parity sanity: mover must always own <= opponent's stone count
    _, _, fs = collect(GENS[0], verify=True)
    bad = sum(n for (nm, no), n in fs.items() if nm > no)
    tot = sum(fs.values())
    print(f"\nmover parity check on gen {GENS[0]}: mover owns MORE stones than "
          f"opponent in {bad}/{tot} positions (must be 0)")
    even = sum(n for (nm, no), n in fs.items() if nm == no)
    print(f"  even-ply positions (equal stone counts) = {even}/{tot} "
          f"({100*even/tot:.0f}%) -- these are exactly the ones ab_heads.py mislabels")

    for M in (13, 14, 15):
        mp = f"{ROOT}/{M}/model.onnx"
        if not os.path.exists(mp):
            continue
        pf = onnx_pred(mp)
        tk = train_keys(M)
        print("\n" + "=" * 92)
        print(f"MODEL {M}  (trained on gens {M-4}..{M-1}, generated gen {M})")
        print("=" * 92)
        print(f"{'gen':>4} {'in-win':>7} | {'WIN1 (true +1)':^36} | "
              f"{'LOSS2 (true -1)':^30}")
        print(f"{'':>4} {'':>7} | {'n':>6} {'meanV':>7} {'MSE':>7} {'sign ok':>8} "
              f"{'MSE|seen':>9} | {'n':>5} {'meanV':>7} {'MSE':>7} {'sign ok':>8}")
        for g in GENS:
            w, l = sets[g]
            inwin = (M - 4) <= g <= (M - 1)
            row = f"{g:>4} {'YES' if inwin else '-':>7} | "
            if len(w):
                v = pf(w)
                seen = np.array([x.tobytes() in tk for x in w])
                ms = float(np.mean((v[seen] - 1) ** 2)) if seen.any() else float("nan")
                row += (f"{len(w):>6} {v.mean():>+7.3f} {float(np.mean((v-1)**2)):>7.3f} "
                        f"{np.mean(v > 0)*100:>7.1f}% {ms:>9.3f} | ")
            else:
                row += f"{'-':>6} {'-':>7} {'-':>7} {'-':>8} {'-':>9} | "
            if len(l):
                v = pf(l)
                row += (f"{len(l):>5} {v.mean():>+7.3f} {float(np.mean((v+1)**2)):>7.3f} "
                        f"{np.mean(v < 0)*100:>7.1f}%")
            else:
                row += f"{'-':>5} {'-':>7} {'-':>7} {'-':>8}"
            print(row)


if __name__ == "__main__":
    main()
