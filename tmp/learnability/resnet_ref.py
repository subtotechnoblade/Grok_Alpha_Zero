"""Production ResNet reference: Connect4/Build_Model.build_model trained the production
way (Muon + Orthograd, lr 3e-4, batch 1024, both heads, sample weights on both) on the
SAME split as bench.py, so its value head lands in the same table.

Difference from a real generation, stated up front: production WARM-STARTS from the
previous generation's weights and runs 10 epochs; this trains from scratch. To avoid
under-crediting it, we run more epochs and report the BEST epoch by val value-MSE
(same selection rule as production's ModelCheckpoint), plus the epoch-10 value for
reference and the whole per-epoch curve.
"""
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf

REPO = "C:/Users/TryHarder/Desktop/Pythoncode/Grok_Zero"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data  # noqa: E402
from Connect4.Build_Model import build_model  # noqa: E402
from Connect4.config import build_config, optimizer_config  # noqa: E402
from Net.Alpha_Loss import Policy_Loss, Value_Loss  # noqa: E402
from Net.Optimizers.muon import Muon  # noqa: E402
from Net.Optimizers.ortho_grad import Orthograd  # noqa: E402

SEED = 0


class Track(tf.keras.callbacks.Callback):
    """Weighted value MSE on every split after each epoch (the harness metric)."""

    def __init__(self, d, splits, target):
        super().__init__()
        self.d, self.splits, self.target = d, splits, target
        self.hist = []

    def on_epoch_end(self, epoch, logs=None):
        row = {"epoch": epoch + 1, "train_loss": float(logs.get("loss", np.nan)),
               "val_loss": float(logs.get("val_loss", np.nan))}
        for s in self.splits + ["train"]:
            p = self.d[s]
            v = self.model.predict(p["X"].astype(np.float32), batch_size=4096, verbose=0)[1].ravel()
            for t in ("blend", "z"):
                row[f"{s}_{t}"] = data.wmse(p[f"y_{t}"], v, p["w"])
        self.hist.append(row)
        print(f"    ep{epoch+1:3d} loss={row['train_loss']:.4f} "
              f"train_{self.target}={row[f'train_{self.target}']:.4f} "
              + " ".join(f"{s}_{self.target}={row[f'{s}_{self.target}']:.4f}" for s in self.splits),
              flush=True)


def train_one(d, target, epochs, use_orthograd, tag, lr=3e-4):
    tf.keras.utils.set_random_seed(SEED)
    splits = [s for s in ("val", "nxt") if s in d]
    tr, va = d["train"], d["val"]

    model = build_model((6, 7, 4), (7,), (build_config, {"use_gumbel": False}, optimizer_config))
    kw = dict(optimizer_config["kwargs"])
    kw["learning_rate"] = lr
    opt = Muon(**kw)
    if use_orthograd:
        opt = Orthograd(opt)
    model.compile(optimizer=opt,
                  loss={"policy": Policy_Loss(dtype="float32"), "value": Value_Loss(dtype="float32")})

    cb = Track(d, splits, target)
    Xtr = tr["X"].astype(np.float32)
    ytr = (tr["policy"].astype(np.float32), tr[f"y_{target}"].astype(np.float32).reshape(-1, 1))
    wtr = (tr["w"].astype(np.float32), tr["w"].astype(np.float32))
    Xva = va["X"].astype(np.float32)
    yva = (va["policy"].astype(np.float32), va[f"y_{target}"].astype(np.float32).reshape(-1, 1))
    wva = (va["w"].astype(np.float32), va["w"].astype(np.float32))

    print(f"  [{tag}] params={model.count_params()} target={target} orthograd={use_orthograd}")
    model.fit(Xtr, ytr, sample_weight=wtr, validation_data=(Xva, yva, wva),
              epochs=epochs, batch_size=1024, verbose=0, callbacks=[cb])

    best = min(cb.hist, key=lambda r: r[f"val_{target}"])
    ep10 = cb.hist[min(9, len(cb.hist) - 1)]
    return {"tag": tag, "target": target, "orthograd": use_orthograd,
            "best_epoch": best["epoch"],
            "val": best[f"val_{target}"],
            "nxt": best.get(f"nxt_{target}"),
            "val_other": best[f"val_{'z' if target=='blend' else 'blend'}"],
            "nxt_other": best.get(f"nxt_{'z' if target=='blend' else 'blend'}"),
            "train_at_best": best[f"train_{target}"],
            "epoch10_val": ep10[f"val_{target}"], "epoch10_nxt": ep10.get(f"nxt_{target}"),
            "epoch10_train": ep10[f"train_{target}"],
            "curve": cb.hist}


def main(run="run2", gen=12, epochs=30):
    t0 = time.time()
    d = data.get(run, gen, seed=SEED)
    out = []
    for target in ("blend", "z"):
        for ortho in (True, False):
            tag = f"ResNet prod{' +OrthoGrad' if ortho else ' (Muon only)'}"
            out.append(train_one(d, target, epochs, ortho, tag))
            print(f"  -> {tag} [{target}] best ep {out[-1]['best_epoch']}: "
                  f"train={out[-1]['train_at_best']:.4f} val={out[-1]['val']:.4f} "
                  f"nxt={out[-1]['nxt']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"resnet_{run}_g{gen}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "run2",
         int(sys.argv[2]) if len(sys.argv) > 2 else 12,
         int(sys.argv[3]) if len(sys.argv) > 3 else 30)
