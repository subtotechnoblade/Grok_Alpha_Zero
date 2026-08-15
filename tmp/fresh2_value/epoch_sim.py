"""Replay the gen-12 training fit (window gens 8-11, warm start from gen-11
weights, muon, 8 epochs) on CPU and record the per-epoch val curves per head.
Question: does val_value_loss bottom out at ~epoch 2 while val_policy_loss keeps
falling, on this window too?

Writes NOTHING outside tmp/fresh2_value/. Reads Grok_Zero_Train read-only.
CPU only, limited threads (live run shares the machine).
"""
import os
import sys
import json
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import h5py as h5
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")          # CPU only, leave GPU to the live run
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)

from Dataloader import dedup_average, Dataloader
from Net.Alpha_Loss import Policy_Loss, Value_Loss, KLD
from Net.Optimizers.muon import Muon
from Connect4.Build_Model import build_model
from Connect4.config import build_config, train_config, optimizer_config, lr_schedule

GENS_NEWEST_FIRST = [11, 10, 9, 8]
TRAIN_PCT = {11: 1.0, 10: 0.9, 9: 0.8, 8: 0.7}
TEST_PCT = {11: 0.10, 10: 0.09, 9: 0.08, 8: 0.07}
WARM_START = "Connect4/Grok_Zero_Train/11/model.weights.h5"
GENERATION = 11                                   # lr_schedule arg used for the real fit


def collect():
    np.random.seed()                              # same as Create_Train_Test_Split
    tr_s, tr_p, tr_v = [], [], []
    te_s, te_p, te_v = [], [], []
    for g in GENS_NEWEST_FIRST:
        with h5.File(f"Connect4/Grok_Zero_Train/{g}/Self_Play_Data.h5", "r") as f:
            n = int(f["game_stats"][2])
            aug = ((len(f.keys()) - 1) // 3) // n
            perm = np.random.permutation(n)
            n_test = int(TEST_PCT[g] * n)
            test_ids = set(perm[:n_test].tolist())
            rest = perm[n_test:]
            train_ids = set(rest[:int(TRAIN_PCT[g] * len(rest))].tolist())
            for gid in range(n):
                if gid in test_ids:
                    S, P, V = te_s, te_p, te_v
                elif gid in train_ids:
                    S, P, V = tr_s, tr_p, tr_v
                else:
                    continue
                for did in range(gid * aug, gid * aug + aug):
                    S += list(f[f"boards_{did}"])
                    P += list(f[f"policies_{did}"])
                    V += list(f[f"values_{did}"])
    return (tr_s, tr_p, tr_v), (te_s, te_p, te_v)


def main():
    t0 = time.time()
    (tr_s, tr_p, tr_v), (te_s, te_p, te_v) = collect()
    print(f"collected: train {len(tr_s)}, test {len(te_s)}  ({time.time()-t0:.0f}s)")

    tr_s, tr_p, tr_v, tr_c = dedup_average(tr_s, tr_p, tr_v)
    te_s, te_p, te_v, te_c = dedup_average(te_s, te_p, te_v)
    tr_w = tr_c * (len(tr_c) / max(1.0, tr_c.sum()))
    te_w = te_c * (len(te_c) / max(1.0, te_c.sum()))
    print(f"dedup: train {len(tr_s)} unique (max w {tr_w.max():.0f}), "
          f"test {len(te_s)} unique (max w {te_w.max():.0f})")

    bs = train_config["train_batch_size"]
    train_dl = Dataloader(tr_s, tr_p, tr_v, bs, weights=tr_w)
    test_dl = Dataloader(te_s, te_p, te_v, bs, weights=te_w)

    configs = [build_config, train_config, optimizer_config]
    model = build_model((6, 7, 4), (7,), configs)
    model.load_weights(WARM_START)

    kw = dict(optimizer_config["kwargs"])
    kw["learning_rate"] = lr_schedule(GENERATION)
    opt = Muon(**kw)
    model.compile(optimizer=opt,
                  loss={"policy": Policy_Loss(dtype="float32"),
                        "value": Value_Loss(dtype="float32")},
                  metrics={"policy": KLD(dtype="float32")})

    print(f"lr {kw['learning_rate']}, batch {bs}, "
          f"{len(train_dl)} steps/epoch x {optimizer_config['train_epochs']} epochs")
    hist = model.fit(train_dl, validation_data=test_dl,
                     epochs=optimizer_config["train_epochs"], verbose=2)

    h = {k: [float(x) for x in v] for k, v in hist.history.items()}
    out = os.path.join(ROOT, "tmp", "fresh2_value", "epoch_sim_history.json")
    with open(out, "w") as f:
        json.dump(h, f, indent=1)
    print(f"\n{'ep':>3} {'train_pol':>10} {'train_val':>10} {'val_pol':>9} "
          f"{'val_val':>9} {'val_loss':>9}")
    for i in range(len(h["loss"])):
        print(f"{i + 1:>3} {h['policy_loss'][i]:>10.4f} {h['value_loss'][i]:>10.4f} "
              f"{h['val_policy_loss'][i]:>9.4f} {h['val_value_loss'][i]:>9.4f} "
              f"{h['val_loss'][i]:>9.4f}")
    for k in ["val_loss", "val_policy_loss", "val_value_loss"]:
        print(f"argmin {k}: epoch {int(np.argmin(h[k])) + 1} ({min(h[k]):.4f})")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
