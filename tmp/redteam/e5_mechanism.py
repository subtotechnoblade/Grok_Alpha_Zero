"""E5: the collapse is generational, not memorisation. What actually changes?

Two candidate mechanisms:
  COVARIATE shift  - gen M's boards sit in a region gens M-4..M-1 never visited.
                     Testable: match on ply and on tactical category; if the gap
                     survives exact matching, composition is not the cause.
  TARGET shift     - the SAME board carries a different value because the outcome
                     depends on the policy that plays it out. Testable on
                     byte-identical, well-pinned positions across generation pairs.

Win-in-1 is the discriminator: its TRUE value is +1 under any policy, so target
shift cannot touch it. If win-in-1 accuracy collapses anyway, the value head has
no policy-independent tactical concept at all.
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


def win1_with_labels(games):
    """win-in-1 states plus their RECORDED blend value and recovered z."""
    X, yb, yz = [], [], []
    for g in games:
        B, V, Z = g["boards"], g["values"][:, 0], g["z"][:, 0]
        for t in range(len(B)):
            st = B[t]
            if win_cols(st[..., -1], -int(st[..., 0].ravel()[0])):
                X.append(st); yb.append(V[t]); yz.append(Z[t])
    z = np.zeros((0, 6, 7, 4), np.int8)
    return (np.asarray(X, np.int8) if X else z,
            np.array(yb), np.array(yz))


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


def ply_of(X):
    return (X[..., -1] != 0).reshape(len(X), -1).sum(1).astype(np.int64)


def main():
    pf = onnx_pred(f"{ROOT}/13/model.onnx")

    # ---------- 1. composition + recorded label of win-in-1 sets ----------
    print("=" * 100)
    print("1. WIN-IN-1 SETS PER GENERATION - composition and the RECORDED label.")
    print("   true value is +1 by the rules. 'rec z' is what self-play actually")
    print("   recorded: if the mover blundered the win away it will be below +1.")
    print("=" * 100)
    print(f"{'gen':>4} {'n':>6} {'mean ply':>9} {'p10':>5} {'p90':>5} "
          f"{'rec blend':>10} {'rec z':>7} {'%z=+1':>7} {'model13 v':>10} {'MSE v,+1':>9}")
    sets = {}
    for g in range(9, 17):
        games = D._read_gen(ROOT, g)
        X, yb, yz = win1_with_labels(games)
        sets[g] = (X, yb, yz)
        p = ply_of(X)
        v = pf(X)
        print(f"{g:>4} {len(X):>6} {p.mean():>9.2f} {np.percentile(p,10):>5.0f} "
              f"{np.percentile(p,90):>5.0f} {yb.mean():>+10.3f} {yz.mean():>+7.3f} "
              f"{100*np.mean(yz > 0.5):>6.1f}% {v.mean():>+10.3f} "
              f"{float(np.mean((v-1)**2)):>9.3f}")

    # ---------- 2. ply-matched comparison ----------
    print("\n" + "=" * 100)
    print("2. PLY-MATCHED: model 13 on win-in-1, gen 12 vs gen 13, same ply bucket.")
    print("   If the collapse survives exact ply matching it is not composition.")
    print("=" * 100)
    X12, _, _ = sets[12]
    X13, _, _ = sets[13]
    v12, v13 = pf(X12), pf(X13)
    p12, p13 = ply_of(X12), ply_of(X13)
    print(f"{'ply band':<12} {'n g12':>6} {'MSE g12':>9} {'signok':>8} | "
          f"{'n g13':>6} {'MSE g13':>9} {'signok':>8}")
    for lo, hi in ((0, 13), (14, 17), (18, 21), (22, 25), (26, 29), (30, 41)):
        a, b = (p12 >= lo) & (p12 <= hi), (p13 >= lo) & (p13 <= hi)
        if a.sum() < 10 or b.sum() < 10:
            continue
        print(f"{f'{lo}-{hi}':<12} {int(a.sum()):>6} "
              f"{float(np.mean((v12[a]-1)**2)):>9.3f} {np.mean(v12[a]>0)*100:>7.1f}% | "
              f"{int(b.sum()):>6} {float(np.mean((v13[b]-1)**2)):>9.3f} "
              f"{np.mean(v13[b]>0)*100:>7.1f}%")

    # ---------- 3. covariate shift: cross-generation board overlap ----------
    print("\n" + "=" * 100)
    print("3. COVARIATE SHIFT - % of gen ROW's loss weight whose board (plane 3 +")
    print("   side to move) also occurs in gen COL. Symmetric-ish if no drift.")
    print("=" * 100)
    packs, keys = {}, {}
    for g in range(9, 17):
        packs[g] = D.pack_games(D._read_gen(ROOT, g))
        keys[g] = {x[..., [0, 3]].tobytes() for x in packs[g]["X"]}
    gens = sorted(packs)
    print(f"{'':>5} " + " ".join(f"g{g:<5}" for g in gens))
    for a in gens:
        P = packs[a]
        kb = [x[..., [0, 3]].tobytes() for x in P["X"]]
        cells = []
        for b in gens:
            m = np.array([k in keys[b] for k in kb])
            cells.append(100 * np.sum(P["w"][m]) / np.sum(P["w"]))
        print(f"g{a:<4} " + " ".join(f"{c:<6.1f}" for c in cells))

    # ---------- 4. target drift matrix ----------
    print("\n" + "=" * 100)
    print("4. TARGET SHIFT - MAE between the blend means of BYTE-IDENTICAL positions")
    print("   in gen ROW vs gen COL, using only positions with count>=20 in both")
    print("   (so occurrence noise contributes <~0.05). Diagonal = split-half null.")
    print("=" * 100)
    maps = {g: {x.tobytes(): (y, c) for x, y, c in
                zip(packs[g]["X"], packs[g]["y_blend"], packs[g]["count"])}
            for g in gens}
    print(f"{'':>5} " + " ".join(f"g{g:<7}" for g in gens))
    for a in gens:
        cells = []
        for b in gens:
            if a == b:
                gg = D._read_gen(ROOT, a)
                rng = np.random.RandomState(0)
                o = rng.permutation(len(gg))
                pa = D.pack_games([gg[i] for i in o[:len(gg)//2]])
                pb = D.pack_games([gg[i] for i in o[len(gg)//2:]])
                mb = {x.tobytes(): (y, c) for x, y, c in
                      zip(pb["X"], pb["y_blend"], pb["count"])}
                dd = [abs(y - mb[x.tobytes()][0]) for x, y, c in
                      zip(pa["X"], pa["y_blend"], pa["count"])
                      if x.tobytes() in mb and c >= 10 and mb[x.tobytes()][1] >= 10]
            else:
                dd = [abs(y - maps[b][k][0]) for k, (y, c) in maps[a].items()
                      if k in maps[b] and c >= 20 and maps[b][k][1] >= 20]
            cells.append(f"{np.mean(dd):.3f}({len(dd)})" if len(dd) >= 5 else "  -    ")
        print(f"g{a:<4} " + " ".join(f"{c:<8}" for c in cells))


if __name__ == "__main__":
    main()
