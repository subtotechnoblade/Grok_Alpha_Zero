"""EXP 6 - confirm the headline with the REAL Connect4 model + production optimiser.

Uses tmp/weight_sweep.build_and_compile(), i.e. Connect4.Build_Model.build_model with
the production build/train/optimizer configs, Muon at lr 3e-4, no OrthoGrad, the real
Policy_Loss/Value_Loss, and the production tuple fit signature
    fit(X, (policy, value), sample_weight=(w, w)).

Guards against the failure mode that made earlier from-scratch arms look identical:
the model is evaluated on train / val / nxt after EVERY epoch, so under-fitting is
visible rather than assumed away.  Production warm-starts each generation from the
previous weights and then runs 10 epochs; from scratch that is not converged, so the
arms here run 40 epochs and the whole trajectory is reported.

Arms (2 seeds each):
  scale    160 / 640 / 1260 games drawn by game from the gens 9-12 window
  fresh    400 games of gen 13 itself, scored on a gen-13 by-game holdout, against
           the 1260-game stale window scored on the same holdout
"""
import json
import os
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np

import common as C

sys.path.insert(0, C.TMP)
os.chdir(C.REPO)

import tensorflow as tf                          # noqa: E402
from weight_sweep import build_and_compile       # noqa: E402

TARGET = sys.argv[1] if len(sys.argv) > 1 else "blend"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
SEEDS = (0, 1)
BATCH = 1024
YK = f"y_{TARGET}"


class Track(tf.keras.callbacks.Callback):
    def __init__(self, pool, evals, every=2):
        super().__init__()
        self.pool, self.evals, self.every = pool, evals, every
        self.hist = []

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every and epoch + 1 != EPOCHS:
            return
        row = {"epoch": epoch + 1, "fit_loss": float(logs.get("loss", np.nan))}
        for nm, p in [("train", self.pool)] + list(self.evals.items()):
            pv = self.model.predict(p["_Xf"], batch_size=4096, verbose=0)[1][:, 0]
            row[nm] = C.wmse(p[YK], pv.astype(np.float64), p["w"])
        self.hist.append(row)
        print("      ep{epoch:>3} fit={fit_loss:.4f} ".format(**row) +
              " ".join(f"{k}={row[k]:.4f}" for k in ["train"] + list(self.evals)),
              flush=True)


def run(pool, evals, seed):
    tf.keras.utils.set_random_seed(seed)
    pool["_Xf"] = pool["X"].astype(np.float32)
    m = build_and_compile()
    cb = Track(pool, evals)
    m.fit(pool["X"].astype(np.float32),
          (pool["policy"].astype(np.float32), pool[YK].astype(np.float32)[:, None]),
          sample_weight=(pool["w"].astype(np.float32), pool["w"].astype(np.float32)),
          batch_size=BATCH, epochs=EPOCHS, verbose=0, shuffle=True, callbacks=[cb])
    tf.keras.backend.clear_session()
    return cb.hist


def main():
    t0 = time.time()
    out = {"target": TARGET, "epochs": EPOCHS, "arms": []}

    # ---------------- part A: scale within the production window
    W = C.D.raw_window("run2", 12, seed=0)
    games, val, nxt = W["train_games"], W["val"], W["nxt"]
    evA = {"val": val, "nxt": nxt}
    for p in evA.values():
        p["_Xf"] = p["X"].astype(np.float32)
    flA = {k: C.floor_of(v, TARGET, "run2", 12) for k, v in evA.items()}
    c0A = C.const0(evA, TARGET)
    print(f"### EXP6 real ResNet, target={TARGET}, {EPOCHS} epochs")
    print(f"  PART A floors val {flA['val']:.4f} nxt {flA['nxt']:.4f} | "
          f"const0 val {c0A['val']:.4f} nxt {c0A['nxt']:.4f}")
    for n in (160, 640, len(games)):
        for seed in SEEDS:
            rng = np.random.RandomState(1000 + seed)
            sel = rng.choice(len(games), size=min(n, len(games)), replace=False)
            pool = C.pack([games[i] for i in sel])
            print(f"\n  [A] {n} games, {len(pool['X'])} uniq, seed {seed}")
            h = run(pool, evA, seed)
            out["arms"].append(dict(part="A", n_games=int(n), seed=seed,
                                    n_uniq=int(len(pool["X"])), hist=h,
                                    floors=flA, const0=c0A))
            C.save(f"exp6_resnet_{TARGET}.json", out)

    # ---------------- part B: fresh-but-small vs stale-but-big
    g13 = C.read_gen("run2", 13)
    tr13, hold13 = C.split_games(g13, 0.2, seed=11)
    evB = {"g13_hold": C.pack(hold13)}
    evB["g13_hold"]["_Xf"] = evB["g13_hold"]["X"].astype(np.float32)
    flB = C.floor_of(evB["g13_hold"], TARGET, "run2", 12)
    c0B = C.const0(evB, TARGET)
    print(f"\n  PART B eval gen13 holdout: floor {flB:.4f} const0 {c0B['g13_hold']:.4f}")
    for name, src, n in (("fresh gen13 x400", tr13, 400),
                         ("stale window 9-12", games, len(games))):
        for seed in SEEDS:
            rng = np.random.RandomState(4000 + seed)
            sel = rng.choice(len(src), size=min(n, len(src)), replace=False)
            pool = C.pack([src[i] for i in sel])
            print(f"\n  [B] {name}: {min(n,len(src))} games, {len(pool['X'])} uniq, seed {seed}")
            h = run(pool, evB, seed)
            out["arms"].append(dict(part="B", arm=name, n_games=int(min(n, len(src))),
                                    seed=seed, n_uniq=int(len(pool["X"])), hist=h,
                                    floor=flB, const0=c0B))
            C.save(f"exp6_resnet_{TARGET}.json", out)

    print(f"\n{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
