"""Trimmed ResNet reference; writes JSON after EACH config so a kill loses nothing.

Configs, chosen to test the two competing explanations directly:
  prod   = Muon + OrthoGrad, lr 3e-4  (exactly Connect4/config.py optimizer_config)
  adam   = plain Adam, lr 1e-3        (does a boring optimizer do better? if yes the
                                       optimizer is implicated; if no it is not)
Usage: resnet_ref2.py <run> <gen> <epochs> <target> <opt>[,<opt>...]
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
HERE = os.path.dirname(os.path.abspath(__file__))


class Track(tf.keras.callbacks.Callback):
    def __init__(self, d, splits, target):
        super().__init__()
        self.d, self.splits, self.target = d, splits, target
        self.hist = []

    def on_epoch_end(self, epoch, logs=None):
        row = {"epoch": epoch + 1, "train_loss": float(logs.get("loss", np.nan))}
        for s in self.splits + ["train"]:
            p = self.d[s]
            v = self.model.predict(p["X"].astype(np.float32), batch_size=8192, verbose=0)[1].ravel()
            for t in ("blend", "z"):
                row[f"{s}_{t}"] = data.wmse(p[f"y_{t}"], v, p["w"])
        self.hist.append(row)
        print(f"    ep{epoch+1:3d} loss={row['train_loss']:.4f} "
              f"train_{self.target}={row[f'train_{self.target}']:.4f} "
              + " ".join(f"{s}_{self.target}={row[f'{s}_{self.target}']:.4f}" for s in self.splits),
              flush=True)


def make_opt(kind):
    if kind == "prod":
        kw = dict(optimizer_config["kwargs"]); kw["learning_rate"] = 3e-4
        return Orthograd(Muon(**kw)), "ResNet prod +OrthoGrad"
    if kind == "muon":
        kw = dict(optimizer_config["kwargs"]); kw["learning_rate"] = 3e-4
        return Muon(**kw), "ResNet prod (Muon only)"
    if kind == "adam":
        return tf.keras.optimizers.Adam(1e-3), "ResNet arch + plain Adam"
    raise ValueError(kind)


def run(run_name, gen, epochs, target, kinds):
    d = data.get(run_name, gen, seed=SEED)
    splits = [s for s in ("val", "nxt") if s in d]
    tr, va = d["train"], d["val"]
    path = os.path.join(HERE, f"resnet2_{run_name}_g{gen}_{target}.json")
    out = json.load(open(path)) if os.path.exists(path) else []

    Xtr = tr["X"].astype(np.float32)
    ytr = (tr["policy"].astype(np.float32), tr[f"y_{target}"].astype(np.float32).reshape(-1, 1))
    wtr = (tr["w"].astype(np.float32), tr["w"].astype(np.float32))
    Xva = va["X"].astype(np.float32)
    yva = (va["policy"].astype(np.float32), va[f"y_{target}"].astype(np.float32).reshape(-1, 1))
    wva = (va["w"].astype(np.float32), va["w"].astype(np.float32))

    for kind in kinds:
        t0 = time.time()
        tf.keras.utils.set_random_seed(SEED)
        model = build_model((6, 7, 4), (7,), (build_config, {"use_gumbel": False}, optimizer_config))
        opt, tag = make_opt(kind)
        model.compile(optimizer=opt, loss={"policy": Policy_Loss(dtype="float32"),
                                           "value": Value_Loss(dtype="float32")})
        print(f"  [{tag}] target={target} params={model.count_params()}", flush=True)
        cb = Track(d, splits, target)
        model.fit(Xtr, ytr, sample_weight=wtr, validation_data=(Xva, yva, wva),
                  epochs=epochs, batch_size=1024, verbose=0, callbacks=[cb])
        best = min(cb.hist, key=lambda r: r[f"val_{target}"])
        out.append({"tag": tag, "target": target, "best_epoch": best["epoch"],
                    "epochs_run": len(cb.hist),
                    "val": best[f"val_{target}"], "nxt": best.get(f"nxt_{target}"),
                    "train_at_best": best[f"train_{target}"],
                    "final_train": cb.hist[-1][f"train_{target}"],
                    "final_val": cb.hist[-1][f"val_{target}"],
                    "final_nxt": cb.hist[-1].get(f"nxt_{target}"),
                    "curve": cb.hist})
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
        print(f"  -> {tag} [{target}] best ep {best['epoch']}: train={best[f'train_{target}']:.4f} "
              f"val={best[f'val_{target}']:.4f} nxt={best.get(f'nxt_{target}'):.4f} "
              f"({time.time()-t0:.0f}s)  [wrote {os.path.basename(path)}]", flush=True)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5].split(","))
