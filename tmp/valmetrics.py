"""ITEM 2 - the value-loss breakdown that production's single number hides.

Production reports ONE weighted MSE against the blended target on the in-window
val split. That number conflates three things:
  * an OPENING bucket whose irreducible floor is 0.4-0.9 (cannot improve) with an
    ENDGAME bucket whose floor is ~0.00 (where all recoverable error lives),
  * positions the net memorised from train with positions it has never seen,
  * the blend target 0.8z+0.2q (or 0.6/0.4), which contains the net's own search
    output and so flatters it ~2x versus predicting the real outcome z.

This splits all three apart. Reusable: pass any predict-fn.

  usage: valmetrics.py [run] [gen]      e.g. valmetrics.py run2 12
         evaluates model (gen+1) - the one trained on the window ending at `gen`
         and therefore held out from BOTH reported splits.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "learnability"))
os.chdir(os.path.dirname(HERE))

import numpy as np

import data as D

BANDS = [(0, 5), (6, 9), (10, 13), (14, 17), (18, 21), (22, 29), (30, 41)]
COARSE = [(0, 9, "opening"), (10, 21, "midgame"), (22, 41, "endgame")]


def ply_of(X):
    """Occupied cells. Plane -1 is the +1/-1 board (Connect4/game.py get_input_state)."""
    p = (X[..., -1] != 0).reshape(len(X), -1).sum(1).astype(np.int64)
    assert p.min() >= 0 and p.max() <= 42, f"ply out of range: {p.min()}..{p.max()}"
    return p


def wmse(y, pred, w):
    return float(np.sum(w * (pred - y) ** 2) / np.sum(w))


def position_floor(ply, count, floors, target):
    """Residual noise of the DEDUPED target, per position.

    sigma2_by_band is PER-SAMPLE variance of z. The training/eval target is the
    mean of `count` draws, so the noise left in it is sigma^2 / count. Comparing a
    deduped loss against raw sigma^2 makes heavily-repeated openings look better
    than physically possible - the floor there is genuinely tiny because the
    average of 1500 games pins the expectation.
    """
    sig = np.full(len(ply), np.nan)
    for k, v in floors[target]["sigma2_by_band"].items():
        lo, hi = (int(x) for x in k.split("-"))
        sig[(ply >= lo) & (ply <= hi)] = v
    sig[np.isnan(sig)] = np.nanmax(sig)
    return sig / np.maximum(count.astype(np.float64), 1.0)


def report(pred_fn, d, floors, splits=("val", "nxt")):
    train_keys = {x.tobytes() for x in d["train"]["X"]}

    for split in splits:
        if split not in d:
            continue
        s = d[split]
        X, w = s["X"], s["w"]
        pred = pred_fn(X)
        ply = ply_of(X)
        seen = np.array([x.tobytes() in train_keys for x in X])

        print(f"\n{'='*86}")
        print(f"SPLIT: {split}   ({len(X)} unique positions, "
              f"{100*np.sum(w[seen])/np.sum(w):.1f}% of loss weight seen in train)")
        print(f"{'='*86}")

        for tname, key in (("blend (what production reports)", "y_blend"),
                           ("pure z (the honest number)", "y_z")):
            y = s[key]
            tgt = "blend" if key == "y_blend" else "z"
            pfloor = position_floor(ply, s["count"], floors, tgt)
            print(f"\n  target = {tname}")
            print(f"    {'bucket':<10} {'n':>6} {'wt%':>6} {'const0':>8} {'floor':>7} "
                  f"{'NET':>8} {'net/floor':>10} | {'unseen only':>12} {'seen only':>10}")
            for lo, hi, name in COARSE:
                m = (ply >= lo) & (ply <= hi)
                if m.sum() == 0:
                    continue
                fl = float(np.sum(w[m] * pfloor[m]) / np.sum(w[m]))
                net = wmse(y[m], pred[m], w[m])
                c0 = wmse(y[m], np.zeros(m.sum()), w[m])
                mu = m & ~seen
                ms = m & seen
                un = wmse(y[mu], pred[mu], w[mu]) if mu.sum() else float("nan")
                se = wmse(y[ms], pred[ms], w[ms]) if ms.sum() else float("nan")
                print(f"    {name:<10} {int(m.sum()):>6} {100*np.sum(w[m])/np.sum(w):>5.1f}% "
                      f"{c0:>8.4f} {fl:>7.4f} {net:>8.4f} {net/fl:>9.1f}x | "
                      f"{un:>12.4f} {se:>10.4f}")
            allnet = wmse(y, pred, w)
            allc0 = wmse(y, np.zeros(len(y)), w)
            aun = wmse(y[~seen], pred[~seen], w[~seen]) if (~seen).sum() else float("nan")
            ase = wmse(y[seen], pred[seen], w[seen]) if seen.sum() else float("nan")
            allfl = float(np.sum(w * pfloor) / np.sum(w))
            print(f"    {'ALL':<10} {len(y):>6} {100.0:>5.1f}% {allc0:>8.4f} "
                  f"{allfl:>7.4f} {allnet:>8.4f} {allnet/allfl:>9.1f}x | "
                  f"{aun:>12.4f} {ase:>10.4f}")


def onnx_pred_fn(path):
    import onnxruntime as rt
    rt.set_default_logger_severity(4)
    sess = rt.InferenceSession(path, providers=["CPUExecutionProvider"])

    def f(X):
        out = []
        for i in range(0, len(X), 2048):
            _, v = sess.run(None, {"inputs": X[i:i + 2048].astype(np.float32)})
            out.append(v[:, 0])
        return np.concatenate(out).astype(np.float64)
    return f


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "run2"
    gen = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    d = D.get(run, gen)
    floors = json.load(open(os.path.join(HERE, "learnability",
                                         f"floorfull_{run}_g{gen}.json")))
    model = f"{D.ROOTS[run]}/{gen+1}/model.onnx"
    print(f"model: {model}  (trained on the window ending at gen {gen}; "
          f"held out from val AND from gen {gen+1})")
    report(onnx_pred_fn(model), d, floors)


if __name__ == "__main__":
    main()
