"""E7: the settling experiment.

State of play:
  * held-out GAMES from in-window gens score like trained games (0.031 vs 0.038)
  * gen-13 win-in-1 collapses (0.632) at EVERY matched ply band
  * the win-in-1 label is +1.000 in every generation -> no target shift
  * board overlap between generations is a flat 12-25%, and g12<->g13 (25.3%) is
    HIGHER than g11<->g12 (15.1%) -> no special covariate shift at the boundary

Two surviving explanations:
  (A) POLICY SHIFT - the net handles novel boards from a policy it trained on but
      not from a newer policy.
  (B) NO CONCEPT   - the net never learned "four in a row" at all. Its success on
      held-out in-window games is proximity: those boards are near-duplicates of
      training boards, while gen-13's are not.

Discriminator: win-in-1 positions from UNIFORM RANDOM PLAY. They belong to no
policy, so (A) makes no prediction about them beyond "off-distribution", while (B)
predicts outright failure. Combined with nearest-neighbour distance to the train
set, this separates "far from training data" from "different policy".

Input tensors are built by replaying moves and are byte-validated against the
stored self-play tensors before being trusted.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tmp", "learnability"))
os.chdir(REPO)

import numpy as np
import data as D

RUN, ROOT = "run2", D.ROOTS["run2"]
ROWS, COLS = 6, 7
DIRS = ((0, 1), (1, 0), (1, 1), (-1, 1))
FIRST = -1          # verified: the first player's discs are -1 in every generation


def landing(b, c):
    for r in range(ROWS - 1, -1, -1):
        if b[r][c] == 0:
            return r
    return -1


def win_cols(b, p):
    out = []
    for c in range(COLS):
        r = landing(b, c)
        if r < 0:
            continue
        for dr, dc in DIRS:
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


def make_state(hist):
    """hist = list of boards, hist[-1] is current. Mirrors game.get_input_state_MCTS:
    ch0 = -(player to move), ch1 = board 2 plies ago, ch2 = 1 ply ago, ch3 = now."""
    ply = int((hist[-1] != 0).sum())
    mover = FIRST if ply % 2 == 0 else -FIRST
    st = np.zeros((6, 7, 4), dtype=np.int8)
    st[..., 0] = -mover
    st[..., 3] = hist[-1]
    st[..., 2] = hist[-2] if len(hist) >= 2 else hist[-1]
    st[..., 1] = hist[-3] if len(hist) >= 3 else st[..., 2]
    return st


def validate():
    """Rebuild the stored tensors from the stored board sequence; must match byte
    for byte, otherwise every random position built below is meaningless.

    NOTE: _read_gen concatenates the `aug` augmentation copies of a game into one
    array, so the sequence must be validated per copy -- looking back across the
    boundary reads the tail of the previous copy. (That concatenation is harmless
    everywhere else: dedup and per-position scoring do not care about order.)"""
    games = D._read_gen(ROOT, 13)[:40]
    bad = tot = 0
    for g in games:
        B = g["boards"]
        aug = 2
        L = len(B) // aug
        for a in range(aug):
            seg = B[a * L:(a + 1) * L]
            for t in range(len(seg)):
                hist = [seg[max(0, t - k)][..., 3] for k in (2, 1, 0)]
                st = make_state(hist)
                tot += 1
                if not np.array_equal(st, seg[t]):
                    bad += 1
    print(f"input-construction validation: {bad}/{tot} rebuilt tensors differ "
          f"from the stored ones (must be 0)")
    return bad == 0


def random_win1(n_target, seed=0, max_tries=400000):
    """Uniform-random legal play; harvest positions where the mover has a win in 1."""
    rng = np.random.RandomState(seed)
    out, tries = [], 0
    while len(out) < n_target and tries < max_tries:
        tries += 1
        board = np.zeros((ROWS, COLS), np.int8)
        hist = [board.copy()]
        mover = FIRST
        for _ in range(42):
            legal = [c for c in range(COLS) if landing(board, c) >= 0]
            if not legal:
                break
            if win_cols(board, mover):
                out.append(make_state(hist[-3:]))
                break
            c = legal[rng.randint(len(legal))]
            r = landing(board, c)
            board = board.copy()
            board[r][c] = mover
            hist.append(board.copy())
            # a completed 4 ends the game
            done = False
            for dr, dc in DIRS:
                k = 1
                for s in (1, -1):
                    rr, cc = r + dr * s, c + dc * s
                    while 0 <= rr < ROWS and 0 <= cc < COLS and board[rr][cc] == mover:
                        k += 1
                        rr += dr * s
                        cc += dc * s
                if k >= 4:
                    done = True
                    break
            if done:
                break
            mover = -mover
    return np.asarray(out, np.int8)


def win1_from_games(games):
    X = []
    for g in games:
        B = g["boards"]
        for t in range(len(B)):
            st = B[t]
            if win_cols(st[..., -1], -int(st[..., 0].ravel()[0])):
                X.append(st)
    return np.asarray(X, np.int8)


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


def onehot(X):
    """current board -> (N,126) one-hot over {-1,0,+1} for fast Hamming via matmul."""
    b = np.asarray(X)[..., 3].reshape(len(X), -1).astype(np.int8)
    return np.concatenate([(b == -1), (b == 0), (b == 1)], 1).astype(np.float32)


def nn_dist(Q, T, chunk=256):
    """min Hamming distance (in cells, 0..42) from each row of Q to any row of T."""
    A, B = onehot(Q), onehot(T)
    out = np.empty(len(Q))
    for i in range(0, len(Q), chunk):
        match = A[i:i + chunk] @ B.T          # cells agreeing
        out[i:i + chunk] = 42.0 - match.max(1)
    return out


def window_heldout(gen=12, n_prev=4, seed=0):
    rng = np.random.RandomState(seed)
    files = [g for g in list(range(gen - n_prev, gen + 1))[1:][::-1] if g >= 0]
    tp, sp = 1.0, 0.1
    held = {}
    for gid in files:
        games = D._read_gen(ROOT, gid)
        order = rng.permutation(len(games))
        n_test = int(sp * len(games))
        held[gid] = [games[g] for g in order[:n_test]]
        tp = max(tp - 0.2, 0.0)
        sp = max(sp - 0.02, 0.0)
    return held


def main():
    if not validate():
        print("ABORT: cannot trust constructed inputs")
        return

    d = D.get(RUN, 12)
    T = d["train"]["X"]
    held = window_heldout(12)

    print("\nbuilding random-play win-in-1 set ...")
    R = random_win1(1200, seed=0)
    R2 = random_win1(1200, seed=1)
    print(f"  random win-in-1: {len(R)} (seed0) / {len(R2)} (seed1), "
          f"mean ply {(R[...,3]!=0).reshape(len(R),-1).sum(1).mean():.1f}")

    sets = {
        "gen12 TRAIN games": win1_from_games(
            [g for g in D._read_gen(ROOT, 12)][:450]),
        "gen12 HELD-OUT games": win1_from_games(held[12]),
        "gen11 HELD-OUT games": win1_from_games(held[11]),
        "gen13 (next gen)": win1_from_games(D._read_gen(ROOT, 13)),
        "RANDOM play seed0": R,
        "RANDOM play seed1": R2,
    }

    print("\n" + "=" * 104)
    print("WIN-IN-1 (true value +1). MSE and sign accuracy, plus how far each set")
    print("sits from the training data (min Hamming distance over the 42 cells).")
    print("=" * 104)
    models = {}
    for M in (13, 14, 15):
        p = f"{ROOT}/{M}/model.onnx"
        if os.path.exists(p):
            models[M] = onnx_pred(p)

    hdr = (f"{'set':<24} {'n':>6} {'ply':>6} {'NNdist':>7} {'%dist=0':>8} | " +
           " | ".join(f"{'m'+str(M)+' MSE':>9} {'sign':>6}" for M in models))
    print(hdr)
    print("-" * len(hdr))
    for name, X in sets.items():
        if len(X) == 0:
            continue
        nd = nn_dist(X, T)
        ply = (X[..., 3] != 0).reshape(len(X), -1).sum(1)
        row = (f"{name:<24} {len(X):>6} {ply.mean():>6.1f} {nd.mean():>7.2f} "
               f"{100*np.mean(nd == 0):>7.1f}% | ")
        cells = []
        for M, pf in models.items():
            v = pf(X)
            cells.append(f"{float(np.mean((v-1)**2)):>9.3f} "
                         f"{np.mean(v > 0)*100:>5.1f}%")
        print(row + " | ".join(cells))

    # ---- ply-matched random vs gen13 vs gen12-heldout, to remove composition ----
    print("\n" + "=" * 104)
    print("PLY-MATCHED (model 13). Removes any depth-composition difference between")
    print("the random set and the self-play sets.")
    print("=" * 104)
    pf = models[13]
    ref = {k: sets[k] for k in ("gen12 HELD-OUT games", "gen13 (next gen)",
                                "RANDOM play seed0")}
    print(f"{'ply band':<10} " + " ".join(f"{k[:18]:>22}" for k in ref))
    for lo, hi in ((8, 15), (16, 21), (22, 27), (28, 33), (34, 41)):
        cells = []
        for k, X in ref.items():
            ply = (X[..., 3] != 0).reshape(len(X), -1).sum(1)
            m = (ply >= lo) & (ply <= hi)
            if m.sum() < 8:
                cells.append(f"{'-':>22}")
                continue
            v = pf(X[m])
            cells.append(f"n={int(m.sum()):<4} "
                         f"{float(np.mean((v-1)**2)):>6.3f} "
                         f"{np.mean(v>0)*100:>5.1f}%")
        print(f"{f'{lo}-{hi}':<10} " + " ".join(f"{c:>22}" for c in cells))


if __name__ == "__main__":
    main()
