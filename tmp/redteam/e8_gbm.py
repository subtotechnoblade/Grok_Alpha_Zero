"""E8: is the GBM really "beating the ResNet as a learner"?

features.hand() hands the model `my_playable3` and `opp_playable3`: the count of
lines holding three of my/your discs whose empty cell is PLAYABLE RIGHT NOW. That
is precisely a win-in-1 / lose-in-1 detector, computed by hand, exactly the fact
E7 shows the ResNet never learned. A win on `nxt` with that feature present says
nothing about the learner.

So: run the GBM on three feature views trained on the identical split and target,
and score them all on the same ground truth the ResNet failed.
    hand+rel  - the claimed configuration (hand features + mover-relative planes)
    rel       - mover-relative planes only: the same information the ResNet gets
    raw       - flattened planes exactly as the network sees them
If `rel`/`raw` collapse to ResNet-like numbers, the gap is the features, not the
learner. 2 seeds each.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tmp", "learnability"))
sys.path.insert(0, HERE)
os.chdir(REPO)

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import data as D
import features as F
from e7_settle import random_win1, win1_from_games

RUN, GEN = "run2", 12
ROOT = D.ROOTS[RUN]


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


def main():
    d = D.get(RUN, GEN)
    tr, va, nx = d["train"], d["val"], d["nxt"]
    W13 = win1_from_games(D._read_gen(ROOT, 13))
    WR0, WR1 = random_win1(1200, seed=0), random_win1(1200, seed=1)

    # does the hand view actually contain a win-in-1 detector?
    h = F.hand(W13)
    idx = F.HAND_NAMES.index("my_playable3")
    hr = F.hand(random_win1(400, seed=7))
    print("sanity: `my_playable3` on positions that ARE win-in-1")
    print(f"  gen13 win-in-1  : mean {h[:, idx].mean():.3f}, "
          f"nonzero {100*np.mean(h[:, idx] > 0):.1f}%")
    print(f"  random win-in-1 : mean {hr[:, idx].mean():.3f}, "
          f"nonzero {100*np.mean(hr[:, idx] > 0):.1f}%")
    print("  -> the feature fires on essentially every win-in-1 position, i.e. the")
    print("     GBM is handed the tactical fact the ResNet has to infer.\n")

    ref = onnx_pred(f"{ROOT}/{GEN+1}/model.onnx")
    print("=" * 104)
    print("Trained on the SAME train split, SAME blend target. Scored on the same")
    print("splits and the same ground truth.")
    print("=" * 104)
    hdr = (f"{'learner':<22} {'sd':>3} {'train':>7} {'val':>7} {'nxt':>7} "
           f"{'win1@g13':>9} {'sign':>6} {'win1@rnd0':>10} {'win1@rnd1':>10} {'sign':>6}")
    print(hdr)
    print("-" * len(hdr))

    for view in ("hand+rel", "hand", "rel", "raw"):
        fn = F.VIEWS[view]
        Xtr = fn(tr["X"])
        for sd in (0, 1):
            m = HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
                l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=15, random_state=sd).fit(
                    Xtr, tr["y_blend"], sample_weight=tr["w"])
            p = lambda X: m.predict(fn(X))
            v13, vr0, vr1 = p(W13), p(WR0), p(WR1)
            print(f"{'GBM ' + view:<22} {sd:>3} "
                  f"{wmse(tr['y_blend'], p(tr['X']), tr['w']):>7.4f} "
                  f"{wmse(va['y_blend'], p(va['X']), va['w']):>7.4f} "
                  f"{wmse(nx['y_blend'], p(nx['X']), nx['w']):>7.4f} "
                  f"{float(np.mean((v13-1)**2)):>9.3f} {np.mean(v13>0)*100:>5.1f}% "
                  f"{float(np.mean((vr0-1)**2)):>10.3f} "
                  f"{float(np.mean((vr1-1)**2)):>10.3f} "
                  f"{np.mean(vr0>0)*100:>5.1f}%", flush=True)

    v13, vr0, vr1 = ref(W13), ref(WR0), ref(WR1)
    print(f"{'ResNet (shipped m13)':<22} {'-':>3} "
          f"{wmse(tr['y_blend'], ref(tr['X']), tr['w']):>7.4f} "
          f"{wmse(va['y_blend'], ref(va['X']), va['w']):>7.4f} "
          f"{wmse(nx['y_blend'], ref(nx['X']), nx['w']):>7.4f} "
          f"{float(np.mean((v13-1)**2)):>9.3f} {np.mean(v13>0)*100:>5.1f}% "
          f"{float(np.mean((vr0-1)**2)):>10.3f} {float(np.mean((vr1-1)**2)):>10.3f} "
          f"{np.mean(vr0>0)*100:>5.1f}%")

    c0 = wmse(nx["y_blend"], np.zeros(len(nx["X"])), nx["w"])
    print(f"\nconst-0 on nxt = {c0:.4f};  const-0 vs win-in-1 ground truth = 1.000")


if __name__ == "__main__":
    main()
