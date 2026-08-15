"""E2: is `nxt` a fair evaluation, or is the split itself broken?

val-endgame UNSEEN = 0.054 while nxt-endgame UNSEEN = 0.811. Both unseen, both
endgame, same model, same target convention. Memorisation cannot explain a 15x
gap between two unseen sets. So either the DATA differs between generations, or
"unseen" is not measuring what it claims.

Tests, in order of how mundane the explanation would be:
  A. Target self-consistency: recover q from value = (1-qw)z + qw*q. If |q|>1 the
     assumed q_weight (or the z recovery) is wrong FOR THAT GENERATION.
  B. Is "unseen" real? Byte-identity includes the 2 history planes. Re-measure
     overlap on the CURRENT BOARD ONLY (plane 3). If val-unseen positions are
     board-identical to train and nxt-unseen are not, the val holdout is fake.
  C. Sign/scale: mean and corr of the net's v against y, per split. A systematic
     flip or offset shows up here and nowhere else.
  D. Cross-generation matrix: model M evaluated on gen G for M in {12,13,14},
     G in 9..16, identical protocol. Locates the cliff -- is it "generations I
     trained on" (overfit) or "generations after me" (shift)?
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


def ply_of(X):
    return (X[..., -1] != 0).reshape(len(X), -1).sum(1).astype(np.int64)


def wmse(y, p, w):
    return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def onnx_pred(path):
    import onnxruntime as rt
    rt.set_default_logger_severity(4)
    s = rt.InferenceSession(path, providers=["CPUExecutionProvider"])

    def f(X):
        o = []
        for i in range(0, len(X), 4096):
            _, v = s.run(None, {"inputs": X[i:i + 4096].astype(np.float32)})
            o.append(v[:, 0])
        return np.concatenate(o).astype(np.float64)
    return f


# ------------------------------------------------------------------ A
def test_A():
    print("=" * 80)
    print("A. TARGET SELF-CONSISTENCY  q = (value - (1-qw)*z)/qw ,  qw assumed 0.2")
    print("   |q| must be <= 1. Also report the fraction of games whose LAST stored")
    print("   value is not in {0, +-1} -- z recovery is invalid for those.")
    print("=" * 80)
    print(f"{'gen':>4} {'games':>6} {'bad_last':>9} {'max|q|':>8} {'%|q|>1.001':>11} "
          f"{'mean z':>8} {'mean val':>9}")
    for gen in range(9, 17):
        games = D._read_gen(ROOT, gen)
        if not games:
            continue
        bad = 0
        qs, zs, vs = [], [], []
        for g in games:
            v = g["values"][:, 0]
            last = float(v[-1])
            if not (abs(last - 1.0) < 1e-4 or abs(last) < 1e-4):
                bad += 1
            z = g["z"][:, 0]
            qs.append((v - 0.8 * z) / 0.2)
            zs.append(z)
            vs.append(v)
        q = np.concatenate(qs)
        print(f"{gen:>4} {len(games):>6} {bad:>9} {np.abs(q).max():>8.3f} "
              f"{100*np.mean(np.abs(q) > 1.001):>10.2f}% "
              f"{np.concatenate(zs).mean():>8.4f} {np.concatenate(vs).mean():>9.4f}")


# ------------------------------------------------------------------ B
def test_B(d):
    print("\n" + "=" * 80)
    print("B. IS 'UNSEEN' REAL?  byte-identity uses all 4 planes (2 are history).")
    print("   Re-measure overlap on the CURRENT BOARD ONLY (plane 3) + side to move.")
    print("=" * 80)
    tr = d["train"]
    full = {x.tobytes() for x in tr["X"]}
    brd = {x[..., [0, 3]].tobytes() for x in tr["X"]}
    print(f"{'split':<5} {'bucket':<9} {'n':>6} {'%wt full-identical':>20} "
          f"{'%wt board-identical':>21}")
    for nm in ("val", "nxt"):
        s = d[nm]
        ply = ply_of(s["X"])
        kf = np.array([x.tobytes() in full for x in s["X"]])
        kb = np.array([x[..., [0, 3]].tobytes() in brd for x in s["X"]])
        for lo, hi, bn in ((0, 9, "opening"), (10, 21, "midgame"),
                           (22, 41, "endgame"), (0, 41, "ALL")):
            m = (ply >= lo) & (ply <= hi)
            if m.sum() == 0:
                continue
            wm = s["w"][m]
            print(f"{nm:<5} {bn:<9} {int(m.sum()):>6} "
                  f"{100*np.sum(wm[kf[m]])/np.sum(wm):>19.1f}% "
                  f"{100*np.sum(wm[kb[m]])/np.sum(wm):>20.1f}%")


# ------------------------------------------------------------------ C
def test_C(d, pf):
    print("\n" + "=" * 80)
    print("C. SIGN / SCALE of the net's output v against the target y (blend).")
    print("=" * 80)
    print(f"{'split':<5} {'bucket':<9} {'mean v':>8} {'mean y':>8} {'sd v':>7} "
          f"{'sd y':>7} {'corr':>7} {'mse(+v)':>9} {'mse(-v)':>9} {'mse(0)':>8}")
    for nm in ("val", "nxt"):
        s = d[nm]
        v = pf(s["X"])
        y, w = s["y_blend"], s["w"]
        ply = ply_of(s["X"])
        for lo, hi, bn in ((0, 9, "opening"), (10, 21, "midgame"),
                           (22, 41, "endgame"), (0, 41, "ALL")):
            m = (ply >= lo) & (ply <= hi)
            if m.sum() < 5:
                continue
            vv, yy, ww = v[m], y[m], w[m]
            mv = np.average(vv, weights=ww)
            my = np.average(yy, weights=ww)
            sv = np.sqrt(np.average((vv - mv) ** 2, weights=ww))
            sy = np.sqrt(np.average((yy - my) ** 2, weights=ww))
            cr = np.average((vv - mv) * (yy - my), weights=ww) / max(1e-12, sv * sy)
            print(f"{nm:<5} {bn:<9} {mv:>+8.4f} {my:>+8.4f} {sv:>7.4f} {sy:>7.4f} "
                  f"{cr:>+7.3f} {wmse(yy, vv, ww):>9.4f} {wmse(yy, -vv, ww):>9.4f} "
                  f"{wmse(yy, np.zeros(m.sum()), ww):>8.4f}")


# ------------------------------------------------------------------ D
def test_D():
    print("\n" + "=" * 80)
    print("D. CROSS-GENERATION MATRIX. Each cell = weighted MSE(blend) of model M on")
    print("   the FULL deduped generation G. Model M trained on gens M-4..M-1.")
    print("=" * 80)
    packs = {}
    for g in range(9, 17):
        games = D._read_gen(ROOT, g)
        if games:
            packs[g] = D.pack_games(games)
    gens = sorted(packs)
    print(f"{'model':>6} | " + " ".join(f"g{g:<6}" for g in gens))
    print("-" * (9 + 8 * len(gens)))
    for M in (11, 12, 13, 14, 15):
        p = f"{ROOT}/{M}/model.onnx"
        if not os.path.exists(p):
            continue
        pf = onnx_pred(p)
        cells = []
        for g in gens:
            s = packs[g]
            cells.append(wmse(s["y_blend"], pf(s["X"]), s["w"]))
        tag = " ".join(f"{c:<7.3f}" for c in cells)
        print(f"{M:>6} | {tag}   <- trained on {M-4}..{M-1}, generated g{M}")
    c0 = [wmse(packs[g]["y_blend"], np.zeros(len(packs[g]["w"])), packs[g]["w"])
          for g in gens]
    print(f"{'const0':>6} | " + " ".join(f"{c:<7.3f}" for c in c0))


def main():
    d = D.get(RUN, 12)
    test_A()
    test_B(d)
    test_C(d, onnx_pred(f"{ROOT}/13/model.onnx"))
    test_D()


if __name__ == "__main__":
    main()
