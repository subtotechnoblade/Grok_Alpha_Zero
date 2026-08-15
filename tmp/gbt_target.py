"""Establish the GBT target the ResNet has to match, on Connect4/backup (10k games, pure z).

The decisive control is FEATURE SET. The earlier GBT win used ~30 HAND features
(playable-3, open-3-on-odd-row, per-column counts). The ResNet only ever sees raw
board planes. So:

  * if GBT[raw cells] also beats the ResNet -> the gap is architecture/optimisation,
    and the target is reachable without touching network inputs.
  * if only GBT[hand] wins -> the gap is REPRESENTATION, and matching it means the
    ResNet must derive threat/parity features from raw boards on its own.

Split by GAME (mirrors of a game stay on the same side), dedup with occurrence
weights, exactly like Dataloader.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "learnability"))
os.chdir(os.path.dirname(HERE))

import numpy as np
import h5py as h5

SRC = "Connect4/backup/Self_Play_Data.h5"
TEST_FRAC = 0.1
SEED = 0


def load():
    with h5.File(SRC, "r") as f:
        n = int(f["game_stats"][2])
        aug = ((len(f.keys()) - 1) // 3) // n
        rng = np.random.RandomState(SEED)
        test_ids = set(rng.permutation(n)[:int(TEST_FRAC * n)].tolist())
        tr_X, tr_y, te_X, te_y = [], [], [], []
        for g in range(n):
            X, y = (te_X, te_y) if g in test_ids else (tr_X, tr_y)
            for a in range(aug):                      # mirrors follow their game
                i = g * aug + a
                X.append(f[f"boards_{i}"][:])
                y.append(f[f"values_{i}"][:][:, 0])
    return (np.concatenate(tr_X), np.concatenate(tr_y),
            np.concatenate(te_X), np.concatenate(te_y))


def dedup(X, y):
    keys = [x.tobytes() for x in X]
    order = {}
    for i, k in enumerate(keys):
        order.setdefault(k, []).append(i)
    idx = np.array([v[0] for v in order.values()])
    cnt = np.array([len(v) for v in order.values()], dtype=np.float64)
    ymean = np.array([y[v].mean() for v in order.values()])
    w = cnt * (len(cnt) / cnt.sum())
    return X[idx], ymean, w, cnt


def wmse(y, p, w):
    return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def ply_of(X):
    return (X[..., -1] != 0).reshape(len(X), -1).sum(1)


def main():
    t0 = time.time()
    trX, trY, teX, teY = load()
    print(f"raw: train {len(trX)}  test {len(teX)}   ({time.time()-t0:.0f}s)")
    trX, trY, trW, _ = dedup(trX, trY)
    teX, teY, teW, teC = dedup(teX, teY)
    print(f"dedup: train {len(trX)} unique  test {len(teX)} unique   "
          f"max train w {trW.max():.0f}")

    ply_te = ply_of(teX)
    import features as F
    sets = {"raw cells (what the ResNet sees)": F.raw,
            "hand (~30 engineered)": F.hand,
            "hand + relative planes": F.hand_rel}

    base = wmse(teY, np.zeros(len(teY)), teW)
    print(f"\nconst-0 baseline: {base:.4f}   (target variance)")
    print(f"\n{'model':<38} {'test wMSE':>10} {'vs const0':>10} "
          f"{'ply0-9':>8} {'ply10-21':>9} {'ply22+':>8}")
    print("-" * 88)

    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError:
        print("sklearn unavailable - cannot run GBT")
        return

    for name, fn in sets.items():
        Ftr, Fte = fn(trX), fn(teX)
        m = HistGradientBoostingRegressor(max_iter=600, max_leaf_nodes=63,
                                          learning_rate=0.06, random_state=0)
        m.fit(Ftr, trY, sample_weight=trW)
        ptr = m.predict(Ftr)
        p = m.predict(Fte)
        tr_tot = wmse(trY, ptr, trW)
        row = [wmse(teY[ply_te <= 9], p[ply_te <= 9], teW[ply_te <= 9]),
               wmse(teY[(ply_te >= 10) & (ply_te <= 21)], p[(ply_te >= 10) & (ply_te <= 21)],
                    teW[(ply_te >= 10) & (ply_te <= 21)]),
               wmse(teY[ply_te >= 22], p[ply_te >= 22], teW[ply_te >= 22])]
        tot = wmse(teY, p, teW)
        print(f"GBT [{name}]".ljust(38) + f"{tr_tot:>9.4f} {tot:>10.4f} "
              f"{1-tot/base:>9.1%} {row[0]:>8.4f} {row[1]:>9.4f} {row[2]:>8.4f}",
              flush=True)

    b0 = [wmse(teY[ply_te <= 9], np.zeros((ply_te <= 9).sum()), teW[ply_te <= 9]),
          wmse(teY[(ply_te >= 10) & (ply_te <= 21)],
               np.zeros(((ply_te >= 10) & (ply_te <= 21)).sum()),
               teW[(ply_te >= 10) & (ply_te <= 21)]),
          wmse(teY[ply_te >= 22], np.zeros((ply_te >= 22).sum()), teW[ply_te >= 22])]
    print(f"{'const-0':<38} {'-':>9} {base:>10.4f} {0.0:>9.1%} "
          f"{b0[0]:>8.4f} {b0[1]:>9.4f} {b0[2]:>8.4f}")
    np.savez(os.path.join(HERE, "gbt_target_split.npz"),
             teX=teX, teY=teY, teW=teW, ply_te=ply_te)
    print(f"\nheld-out split saved for the ResNet comparison. {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
