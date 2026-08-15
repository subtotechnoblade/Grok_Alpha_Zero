"""LR sweep for the CURRENT stack, production-faithful.

Differences from tmp/exp_c_seeds_lr.py / exp_d_lr_curves.py, all of which can
move the optimum: fixed-reshape muon only, current 64-filter architecture,
warm start from the previous generation's weights (as every real generation
does), the REAL gens 6-9 window at full size, and - new today - dedup
occurrence sample-weights, which concentrate the opening's gradient into rare
heavy batches.

Metric: min weighted val_loss over 8 epochs = what ModelCheckpoint keeps.
"""
import os
import sys
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import tensorflow as tf

# same as pipeline.Train_NN: without growth TF grabs all VRAM up front.
# No-op on CPU (windows), active on the WSL GPU run
for _dev in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_dev, True)

TAG = sys.argv[1] if len(sys.argv) > 1 else "cpu"

from Connect4.game import Connect4
from Connect4.config import train_config, build_config, optimizer_config
from Connect4.Build_Model import build_model
from Dataloader import Create_Dataset
from Net.Alpha_Loss import Policy_Loss, Value_Loss
from Net.Optimizers.muon import Muon

# optional argv[2] = comma separated lrs, e.g. "3e-4,1e-3,3e-3"
LRS = ([float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2
       else [1e-3, 3e-3, 1e-2])
SEEDS = 3
EPOCHS = optimizer_config["train_epochs"]          # 8, as production
WARM = "Connect4/Grok_Zero_Train/9/model.weights.h5"

# one dataset for every run - the split is random per call, so build it once
train_dl, test_dl = Create_Dataset(
    "Connect4/Grok_Zero_Train",
    num_previous_generations=train_config["num_previous_generations"],
    train_batch_size=train_config["train_batch_size"],
    test_batch_size=train_config["test_batch_size"],
    train_percent=train_config["train_percent"],
    train_decay=train_config["train_decay"],
    test_percent=train_config["test_percent"],
    test_decay=train_config["test_decay"],
    dedup_average=train_config["dedup_average"])

game = Connect4()
configs = (build_config, train_config, optimizer_config)
results = {}

for lr in LRS:
    best_vals, final_trains = [], []
    for seed in range(SEEDS):
        tf.keras.backend.clear_session()
        model = build_model(game.get_input_state().shape, game.policy_shape, configs)
        model.load_weights(WARM)                   # warm start, like every real gen

        kwargs = dict(optimizer_config["kwargs"])  # copy: never mutate the config
        kwargs["learning_rate"] = lr
        model.compile(optimizer=Muon(**kwargs),
                      loss={"policy": Policy_Loss(dtype="float32"),
                            "value": Value_Loss(dtype="float32")})
        hist = model.fit(train_dl, validation_data=test_dl,
                         epochs=EPOCHS, verbose=0)
        best_vals.append(float(np.min(hist.history["val_loss"])))
        final_trains.append(float(hist.history["loss"][-1]))
        print(f"lr={lr:g} seed={seed}: best_val={best_vals[-1]:.4f} "
              f"final_train={final_trains[-1]:.4f}", flush=True)
    results[f"{lr:g}"] = {"best_val": best_vals, "final_train": final_trains}

with open(f"tmp/results_lr_current_{TAG}.json", "w") as f:
    json.dump(results, f, indent=1)

print("\n" + "=" * 58)
print(f"{'lr':>8} {'best val (mean +/- std)':>26} {'final train':>13}")
for lr, r in results.items():
    bv = np.array(r["best_val"])
    print(f"{lr:>8} {bv.mean():>16.4f} +/- {bv.std(ddof=1):<7.4f} "
          f"{np.mean(r['final_train']):>10.4f}")
