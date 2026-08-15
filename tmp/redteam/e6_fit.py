"""E6: does MODEL FIT change any of the conclusions?

The standing objection to claims 2/3/4 is that the swept models were trained 10
epochs from scratch and may never have reached production fit, so "all arms equal"
could just be "all arms equally unlearned".

Reference point, measured not assumed: the SHIPPED run2 model 13 scores
    train(blend) 0.0667   val 0.0927   nxt 0.6447
so "production fit" for this window means train wmse ~0.067.

Arms here report TRAIN loss alongside the held-out numbers, so fit is verifiable
rather than assumed. Two seeds per arm. Judged on three metrics:
  val        - in-window held-out games (the production metric)
  nxt        - next generation (the deployment metric under test)
  win1@g13   - ground-truth MSE against +1 on gen-13 win-in-1 positions.
               Needs no floor and no distributional assumption.

  usage: e6_fit.py [epochs_list] [schemes]   e.g. e6_fit.py 10,40,120 count,uniform
"""
import json
import os
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tmp", "learnability"))
sys.path.insert(0, os.path.join(REPO, "tmp"))
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np
import data as D

RUN, GEN = "run2", 12
ROOT = D.ROOTS[RUN]
ROWS, COLS = 6, 7
DIRS = ((0, 1), (1, 0), (1, 1), (-1, 1))

EPOCHS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "10,40,120").split(",")]
SCHEMES = (sys.argv[2] if len(sys.argv) > 2 else "count,uniform,cap20").split(",")
SEEDS = [0, 1]


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


def win1_states(games):
    X = []
    for g in games:
        B = g["boards"]
        for t in range(len(B)):
            st = B[t]
            if win_cols(st[..., -1], -int(st[..., 0].ravel()[0])):
                X.append(st)
    return np.asarray(X, np.int8)


def wmse(y, p, w):
    return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def norm1(w):
    return w * (len(w) / w.sum())


def main():
    import tensorflow as tf
    # weight_sweep parses sys.argv at import time; hide our own args from it so we
    # reuse the REAL production model + optimizer rather than re-implementing them.
    _argv, sys.argv = sys.argv, sys.argv[:1]
    from weight_sweep import build_and_compile
    sys.argv = _argv
    sys.path.insert(0, HERE)
    from e7_settle import random_win1

    d = D.get(RUN, GEN)
    tr, va, nx = d["train"], d["val"], d["nxt"]
    W13 = win1_states(D._read_gen(ROOT, 13)).astype(np.float32)
    WRND = random_win1(1200, seed=0).astype(np.float32)
    print(f"train {len(tr['X'])} unique | val {len(va['X'])} | nxt {len(nx['X'])} "
          f"| win1@g13 {len(W13)} | win1@random {len(WRND)}")
    print("reference (shipped model 13): train 0.0667  val 0.0927  nxt 0.6447  "
          "win1@g13 MSE 0.632  win1@random MSE 1.025\n")

    Xtr = tr["X"].astype(np.float32)
    Ptr = tr["policy"].astype(np.float32)
    Vtr = tr["y_blend"].astype(np.float32)[:, None]
    n = tr["count"].astype(np.float64)
    WS = {"count": norm1(n), "uniform": norm1(np.ones_like(n)),
          "sqrt": norm1(np.sqrt(n)), "cap20": norm1(np.minimum(n, 20.0))}

    Xva, Xnx = va["X"].astype(np.float32), nx["X"].astype(np.float32)
    out = []
    hdr = (f"{'scheme':<8} {'ep':>4} {'sd':>3} {'TRAIN':>8} {'val':>8} {'nxt':>8} "
           f"{'nxt/const0':>11} {'win1@g13':>9} {'sign':>6} {'win1@rnd':>9} "
           f"{'sign':>6} {'sec':>6}")
    print(hdr)
    print("-" * len(hdr))
    c0nx = wmse(nx["y_blend"], np.zeros(len(Xnx)), nx["w"])

    for scheme in SCHEMES:
        w = WS[scheme].astype(np.float32)
        for ep in EPOCHS:
            for sd in SEEDS:
                tf.keras.utils.set_random_seed(sd)
                model = build_and_compile()
                t0 = time.time()
                model.fit(Xtr, (Ptr, Vtr), sample_weight=(w, w), batch_size=1024,
                          epochs=ep, verbose=0, shuffle=True)
                dt = time.time() - t0
                pv = lambda X: model.predict(X, batch_size=4096, verbose=0)[1][:, 0] \
                    .astype(np.float64)
                vtr, vva, vnx = pv(Xtr), pv(Xva), pv(Xnx)
                vw, vr = pv(W13), pv(WRND)
                r = dict(scheme=scheme, epochs=ep, seed=sd,
                         train=wmse(tr["y_blend"], vtr, tr["w"]),
                         val=wmse(va["y_blend"], vva, va["w"]),
                         nxt=wmse(nx["y_blend"], vnx, nx["w"]),
                         win1=float(np.mean((vw - 1) ** 2)),
                         signok=float(np.mean(vw > 0)),
                         win1r=float(np.mean((vr - 1) ** 2)),
                         signokr=float(np.mean(vr > 0)), sec=dt)
                out.append(r)
                print(f"{scheme:<8} {ep:>4} {sd:>3} {r['train']:>8.4f} "
                      f"{r['val']:>8.4f} {r['nxt']:>8.4f} {r['nxt']/c0nx:>10.2f}x "
                      f"{r['win1']:>9.3f} {r['signok']*100:>5.1f}% "
                      f"{r['win1r']:>9.3f} {r['signokr']*100:>5.1f}% {dt:>6.0f}",
                      flush=True)
                del model
                tf.keras.backend.clear_session()

    with open(os.path.join(HERE, "e6_fit.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nconst-0 on nxt = {c0nx:.4f}")


if __name__ == "__main__":
    main()
