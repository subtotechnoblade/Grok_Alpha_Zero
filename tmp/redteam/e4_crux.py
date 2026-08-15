"""E4: THE CRUX. Memorisation or distribution shift?

E3 showed model M is near-perfect on win-in-1 in gens M-4..M-1 and fails on gen M.
But those two sets differ in TWO ways at once: train membership AND generation.
valmetrics' val split says the net is fine (0.054 endgame) on UNSEEN positions from
gens M-4..M-1, which points at shift, not memorisation. E3's "MSE|seen" inside gen M
points the other way. Exactly one of these can be right.

The control that separates them: production splits BY GAME, so ~10% of the games in
each in-window generation were never trained on. Those games are
    * from a generation the model trained on  (same distribution)
    * never seen by the model                 (no memorisation possible)

  gen12 train-games  vs  gen12 HELD-OUT games  vs  gen13 (all unseen, next gen)

If held-out gen12 ~= train gen12 and gen13 collapses -> distribution shift is real.
If held-out gen12 collapses too -> it is memorisation and the val split is fake.

Replicated over models 13/14/15 (independent windows = independent replications).
Also reports n_seen so E3's MSE|seen is interpretable, and measures genuine target
drift for byte-identical positions using only well-pinned (high-count) means.
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


def gt_from_games(games):
    """-> X_win (true +1), X_loss (true -1) for a list of per-game dicts."""
    win, loss = [], []
    for g in games:
        B = g["boards"]
        for t in range(len(B)):
            st = B[t]
            board = st[..., -1]
            mover = -int(st[..., 0].ravel()[0])
            if win_cols(board, mover):
                win.append(st)
            elif len(win_cols(board, -mover)) >= 2:
                loss.append(st)
    z = np.zeros((0, 6, 7, 4), np.int8)
    return (np.asarray(win, np.int8) if win else z,
            np.asarray(loss, np.int8) if loss else z)


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


def window_games(gen, n_prev=4, seed=0):
    """Replicates build()'s per-generation game-level split EXACTLY.
    -> {gid: (train_games, heldout_games)} for the 4 in-window generations."""
    rng = np.random.RandomState(seed)
    files = [g for g in list(range(gen - n_prev, gen + 1))[1:][::-1] if g >= 0]
    tp, sp = 1.0, 0.1
    out = {}
    for gid in files:
        games = D._read_gen(ROOT, gid)
        order = rng.permutation(len(games))
        n_test = int(sp * len(games))
        test_ids = set(order[:n_test].tolist())
        rest = order[n_test:]
        train_ids = set(rest[:int(tp * len(rest))].tolist())
        out[gid] = ([games[g] for g in range(len(games)) if g in train_ids],
                    [games[g] for g in range(len(games)) if g in test_ids])
        tp = max(tp - 0.2, 0.0)
        sp = max(sp - 0.02, 0.0)
    return out


def main():
    print("=" * 96)
    print("CRUX: within-generation, held-out-GAME ground truth.")
    print("  'train games'   = games the model actually fitted")
    print("  'held-out games'= games from the SAME generation the model never saw")
    print("  'next gen'      = the generation the model itself produced")
    print("=" * 96)

    for M in (13, 14, 15):
        mp = f"{ROOT}/{M}/model.onnx"
        if not os.path.exists(mp):
            continue
        pf = onnx_pred(mp)
        wg = window_games(M - 1)
        nxt_games = D._read_gen(ROOT, M)
        print(f"\nMODEL {M}  (window {M-4}..{M-1}, produced gen {M})")
        print(f"  {'source':<34} {'n_win':>6} {'MSE(win,+1)':>12} {'sign ok':>8} "
              f"{'n_loss':>7} {'MSE(loss,-1)':>13}")

        def row(tag, games):
            w, l = gt_from_games(games)
            if len(w) == 0:
                return
            v = pf(w)
            s = f"  {tag:<34} {len(w):>6} {float(np.mean((v-1)**2)):>12.3f} " \
                f"{np.mean(v > 0)*100:>7.1f}%"
            if len(l):
                vl = pf(l)
                s += f" {len(l):>7} {float(np.mean((vl+1)**2)):>13.3f}"
            else:
                s += f" {len(l):>7} {'-':>13}"
            print(s)

        for gid in sorted(wg, reverse=True):
            tr_g, ho_g = wg[gid]
            row(f"gen{gid} TRAIN games (n={len(tr_g)})", tr_g)
            row(f"gen{gid} HELD-OUT games (n={len(ho_g)})", ho_g)
        row(f"gen{M} next gen, ALL games (n={len(nxt_games)})", nxt_games)

    # ---------------- how big is E3's "seen" subset really? ----------------
    print("\n" + "=" * 96)
    print("Interpretability of E3's MSE|seen: how many gen-M win-in-1 positions are")
    print("actually byte-identical to something in model M's train set?")
    print("=" * 96)
    for M in (13, 14, 15):
        d = D.get(RUN, M - 1, verbose=False) if False else D.get(RUN, M - 1)
        tk = {x.tobytes() for x in d["train"]["X"]}
        w, _ = gt_from_games(D._read_gen(ROOT, M))
        seen = np.array([x.tobytes() in tk for x in w])
        print(f"  model {M} on gen {M}: {int(seen.sum())} of {len(w)} win-in-1 "
              f"positions seen in train ({100*seen.mean():.2f}%)")

    # ---------------- genuine target drift, noise-free ----------------
    print("\n" + "=" * 96)
    print("TARGET DRIFT for BYTE-IDENTICAL positions, restricted to positions whose")
    print("mean is well pinned on BOTH sides (count >= 30 each), so the difference")
    print("cannot be occurrence noise. Compares the pooled window gens 9-12 against")
    print("gen 13, and as a null, gen 11 against gen 12 (both inside the window).")
    print("=" * 96)

    def packed(gens):
        gg = []
        for g in gens:
            gg += D._read_gen(ROOT, g)
        return D.pack_games(gg)

    def drift(a_gens, b_gens, thresh=30):
        A, B = packed(a_gens), packed(b_gens)
        ma = {x.tobytes(): (y, c) for x, y, c in zip(A["X"], A["y_blend"], A["count"])}
        ya, yb, ca, cb = [], [], [], []
        for x, y, c in zip(B["X"], B["y_blend"], B["count"]):
            k = x.tobytes()
            if k in ma and ma[k][1] >= thresh and c >= thresh:
                ya.append(ma[k][0]); ca.append(ma[k][1]); yb.append(y); cb.append(c)
        if not ya:
            return None
        ya, yb = np.array(ya), np.array(yb)
        ca, cb = np.array(ca, float), np.array(cb, float)
        return len(ya), float(np.mean((ya - yb) ** 2)), float(np.mean(np.abs(ya - yb))), \
            float(np.mean(ca)), float(np.mean(cb))

    for tag, a, b in (("gens 9-12  ->  gen 13", [9, 10, 11, 12], [13]),
                      ("gens 9-11  ->  gen 12  (null, in-window)", [9, 10, 11], [12]),
                      ("gens 9,10  ->  gen 11  (null, in-window)", [9, 10], [11])):
        r = drift(a, b)
        if r:
            n, mse, mae, cA, cB = r
            print(f"  {tag:<42} n={n:>4}  MSE={mse:.4f}  MAE={mae:.4f}  "
                  f"(mean counts {cA:.0f} / {cB:.0f})")


if __name__ == "__main__":
    main()
