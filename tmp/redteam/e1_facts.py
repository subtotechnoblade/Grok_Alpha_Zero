"""E1: independently re-derive the val/nxt numbers, and build an ORACLE that needs
no sigma^2 estimate at all.

Three things this settles:
  1. Are the headline numbers reproducible (val 0.0927 / nxt 0.6447, const0 0.5346)?
  2. Are val and nxt structurally comparable (count distribution, train overlap,
     dedup denominator)?
  3. What does a predictor that CANNOT be blamed on the net score on nxt?
       - oracle_train : for a nxt position also in train, predict train's deduped mean.
       - split-half   : split gen13's GAMES in two, dedup each half, and compare the
                        two independent means of the same position. E[(mA-mB)^2] is a
                        direct, distribution-free measurement of the label noise that
                        remains in the deduped target -- exactly what the sigma^2/count
                        floor is trying to estimate, but measured instead of modelled.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tmp", "learnability"))
os.chdir(REPO)

import numpy as np
import data as D

RUN, GEN = "run2", 12


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
        for i in range(0, len(X), 2048):
            _, v = s.run(None, {"inputs": X[i:i + 2048].astype(np.float32)})
            o.append(v[:, 0])
        return np.concatenate(o).astype(np.float64)
    return f


def main():
    d = D.get(RUN, GEN)
    floors = json.load(open(os.path.join(REPO, "tmp", "learnability",
                                         f"floorfull_{RUN}_g{GEN}.json")))
    tr, va, nx = d["train"], d["val"], d["nxt"]

    # ---- plane-0 sanity: is the board perspective-normalised? ----
    p0 = np.unique(nx["X"][:200, ..., 0])
    print(f"plane0 (current_player) uniques over 200 nxt states: {p0}")
    print(f"boards shape {nx['X'].shape}\n")

    # ---- structural comparison of the two splits ----
    trk = {x.tobytes() for x in tr["X"]}
    print(f"{'split':<6} {'uniq':>7} {'raw':>8} {'raw/uniq':>9} {'cnt=1':>7} "
          f"{'cnt>=10':>8} {'mean cnt':>9} {'%wt in train':>13} {'mean ply':>9}")
    for nm, s in (("train", tr), ("val", va), ("nxt", nx)):
        seen = np.array([x.tobytes() in trk for x in s["X"]])
        c = s["count"]
        raw = int(c.sum())
        print(f"{nm:<6} {len(c):>7} {raw:>8} {raw/len(c):>9.2f} "
              f"{np.mean(c == 1)*100:>6.1f}% {np.mean(c >= 10)*100:>7.1f}% "
              f"{c.mean():>9.2f} {100*np.sum(s['w'][seen])/np.sum(s['w']):>12.1f}% "
              f"{np.average(ply_of(s['X']), weights=s['w']):>9.2f}")

    # ---- reproduce the headline numbers ----
    mpath = f"{D.ROOTS[RUN]}/{GEN+1}/model.onnx"
    pf = onnx_pred(mpath)
    print(f"\nmodel: {mpath}")
    print(f"\n{'split':<6} {'target':<7} {'const0':>8} {'NET':>8} {'floor':>8} "
          f"{'net/floor':>10} {'net/const0':>11}")
    res = {}
    for nm, s in (("val", va), ("nxt", nx)):
        pred = pf(s["X"])
        ply = ply_of(s["X"])
        for tgt, key in (("blend", "y_blend"), ("z", "y_z")):
            y, w = s[key], s["w"]
            sig = np.full(len(ply), np.nan)
            for k, v in floors[tgt]["sigma2_by_band"].items():
                lo, hi = (int(x) for x in k.split("-"))
                sig[(ply >= lo) & (ply <= hi)] = v
            sig[np.isnan(sig)] = np.nanmax(sig)
            fl = float(np.sum(w * (sig / np.maximum(s["count"], 1))) / np.sum(w))
            net = wmse(y, pred, w)
            c0 = wmse(y, np.zeros(len(y)), w)
            res[(nm, tgt)] = (c0, net, fl)
            print(f"{nm:<6} {tgt:<7} {c0:>8.4f} {net:>8.4f} {fl:>8.4f} "
                  f"{net/fl:>9.1f}x {net/c0:>10.2f}x")

    # ---------------- ORACLE 1: train-mean transfer -----------------
    print("\n" + "=" * 78)
    print("ORACLE 1 - predict nxt/val with the TRAIN deduped mean (no net involved).")
    print("If a memorising oracle also scores badly, the target is noisy/shifted,")
    print("not the network. Restricted to the overlapping positions only.")
    print("=" * 78)
    for tgt in ("blend", "z"):
        tmap = {}
        for x, y in zip(tr["X"], tr[f"y_{tgt}"]):
            tmap[x.tobytes()] = y
        print(f"\n  target={tgt}")
        print(f"    {'split':<6} {'n_overlap':>10} {'wt%':>7} {'oracle':>8} "
              f"{'NET':>8} {'const0':>8}")
        for nm, s in (("val", va), ("nxt", nx)):
            keys = [x.tobytes() for x in s["X"]]
            m = np.array([k in tmap for k in keys])
            if m.sum() == 0:
                print(f"    {nm:<6} {'0':>10}   (no overlap)")
                continue
            orc = np.array([tmap[k] for k, ok in zip(keys, m) if ok])
            y, w = s[f"y_{tgt}"][m], s["w"][m]
            pred = pf(s["X"][m])
            print(f"    {nm:<6} {int(m.sum()):>10} "
                  f"{100*np.sum(s['w'][m])/np.sum(s['w']):>6.1f}% "
                  f"{wmse(y, orc, w):>8.4f} {wmse(y, pred, w):>8.4f} "
                  f"{wmse(y, np.zeros(m.sum()), w):>8.4f}")

    # ------------- ORACLE 2: split-half measured noise ---------------
    print("\n" + "=" * 78)
    print("ORACLE 2 - measured label noise in the DEDUPED nxt target.")
    print("Split gen13's games into halves A/B, dedup each, keep positions present in")
    print("both. E[(mA-mB)^2] = Var(mA)+Var(mB) ~= 2 * floor_at_half_count.")
    print("So floor at the FULL count ~= E[(mA-mB)^2]/2 * (cA+cB)/(cA*4/(cA+cB)) ...")
    print("-- reported directly as the unbiased per-position combination below.")
    print("=" * 78)
    games = D._read_gen(D.ROOTS[RUN], GEN + 1)
    rng = np.random.RandomState(0)
    order = rng.permutation(len(games))
    half = len(games) // 2
    ga = [games[i] for i in order[:half]]
    gb = [games[i] for i in order[half:]]
    pa, pb = D.pack_games(ga), D.pack_games(gb)
    for tgt in ("blend", "z"):
        ma = {x.tobytes(): (y, c) for x, y, c in
              zip(pa["X"], pa[f"y_{tgt}"], pa["count"])}
        rows = [(ma[k][0], ma[k][1], y, c) for k, y, c in
                zip((x.tobytes() for x in pb["X"]), pb[f"y_{tgt}"], pb["count"])
                if k in ma]
        if not rows:
            continue
        yA = np.array([r[0] for r in rows], dtype=np.float64)
        cA = np.array([r[1] for r in rows], dtype=np.float64)
        yB = np.array([r[2] for r in rows], dtype=np.float64)
        cB = np.array([r[3] for r in rows], dtype=np.float64)
        # E[(mA-mB)^2] = s2/cA + s2/cB  ->  unbiased s2 per position
        s2 = (yA - yB) ** 2 / (1.0 / cA + 1.0 / cB)
        cT = cA + cB
        # weight by the full-data occurrence count, as the loss does
        wT = cT
        floor_meas = float(np.sum(wT * (s2 / cT)) / np.sum(wT))
        print(f"  target={tgt}: {len(rows)} positions in both halves "
              f"(mean full count {cT.mean():.1f})")
        print(f"    measured sigma^2 (occurrence-weighted) = "
              f"{float(np.sum(wT*s2)/np.sum(wT)):.4f}")
        print(f"    measured floor sigma^2/count           = {floor_meas:.4f}")
        json.dump({"tgt": tgt, "floor_meas": floor_meas}, open(os.devnull, "w"))


if __name__ == "__main__":
    main()
