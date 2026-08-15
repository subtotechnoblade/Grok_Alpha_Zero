"""ITEM 6: controlled learnability test.

Real Connect4 boards (drawn from the run3 archive, so the input distribution is
the real one), real model, real compile path (Muon + Orthograd + Policy_Loss /
Value_Loss + weighted_metrics), real Dataloader, real sample-weight distribution.

ONLY the targets are replaced by a noiseless deterministic function of the board:
    value  = (#discs of player +1) / 10.5 - 1        in [-1, 1]
    policy = uniform over the legal (non-full) columns
Train and test states are DISJOINT (no state appears in both), so val_loss is an
honest generalisation measure here.

If the pipeline cannot drive train AND val loss to ~0 on this, the defect is in
the pipeline/optimizer, not the data.
"""
import os, sys, time, json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
REPO = r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero"
sys.path.insert(0, REPO)

import numpy as np, h5py as h5, tensorflow as tf

N_TRAIN = int(os.environ.get("N_TRAIN", 12000))
N_TEST = int(os.environ.get("N_TEST", 3000))
EPOCHS = int(os.environ.get("EPOCHS", 12))

ARCHIVE = os.path.join(REPO, "tmp", "run3_archive_gen0-15_bottleneck_preweightfix")

# ---------------------------------------------------------------- real boards
states = []
for gen in ["15", "14", "13", "12", "11", "10", "9", "8"]:
    try:
        with h5.File(os.path.join(ARCHIVE, gen, "Self_Play_Data.h5"), "r") as f:
            nd = (len(f.keys()) - 1) // 3
            for ds in range(nd):
                states.append(f[f"boards_{ds}"][:])
    except OSError as e:
        print("h5 read failed for gen", gen, e)
    n = len(np.unique(np.concatenate(states, 0).reshape(-1, 6 * 7 * 4), axis=0))
    print("  after gen", gen, "unique states:", n, flush=True)
    if n > 1.3 * (N_TRAIN + N_TEST):
        break
S = np.concatenate(states, 0)
S = np.unique(S.reshape(len(S), -1), axis=0).reshape(-1, 6, 7, 4).astype(np.int8)
rng = np.random.RandomState(0)
rng.shuffle(S)
S = S[:N_TRAIN + N_TEST]
print("unique real states available:", len(S))
assert len(S) >= N_TRAIN + N_TEST, "not enough unique states"


def targets(S):
    board = S[:, :, :, 3]
    n_plus = (board == 1).sum((1, 2)).astype(np.float32)
    v = (n_plus / 10.5 - 1.0).astype(np.float32).reshape(-1, 1)
    legal = ((board != 0).sum(1) < 6).astype(np.float32)     # (N,7)
    p = legal / legal.sum(1, keepdims=True)
    return p.astype(np.float32), v


P, V = targets(S)
print("value target range [%.3f, %.3f] std %.4f" % (V.min(), V.max(), V.std()))

tr_s, te_s = S[:N_TRAIN], S[N_TRAIN:]
tr_p, te_p = P[:N_TRAIN], P[N_TRAIN:]
tr_v, te_v = V[:N_TRAIN], V[N_TRAIN:]
ov = len(set(map(bytes, tr_s.reshape(N_TRAIN, -1))) &
         set(map(bytes, te_s.reshape(len(te_s), -1))))
print("train/test state overlap:", ov, "(must be 0)")

# real-shaped sample weights: heavy tail with mean 1, like the dedup counts
c = rng.pareto(1.2, N_TRAIN) + 1.0
tr_w = (c * (N_TRAIN / c.sum())).astype(np.float32)
c2 = rng.pareto(1.2, len(te_s)) + 1.0
te_w = (c2 * (len(te_s) / c2.sum())).astype(np.float32)
print("train weight mean %.4f max %.1f" % (tr_w.mean(), tr_w.max()))

from Connect4.game import Connect4
from Connect4.config import build_config, train_config, optimizer_config, lr_schedule
from Connect4.Build_Model import build_model
from Dataloader import Dataloader
import Train as TrainMod

game = Connect4()
configs = (build_config, train_config, optimizer_config)
model = build_model(game.get_input_state().shape, game.policy_shape, configs)

train_dl = Dataloader(list(tr_s), list(tr_p), list(tr_v),
                      train_config["train_batch_size"], weights=tr_w)
test_dl = Dataloader(list(te_s), list(te_p), list(te_v),
                     train_config["train_batch_size"], weights=te_w)
print("batches/epoch:", len(train_dl), "val batches:", len(test_dl))

opt_cfg = dict(optimizer_config)
opt_cfg["train_epochs"] = EPOCHS
cfgs = (build_config, train_config, opt_cfg)

t0 = time.time()


class Timer(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("  epoch %2d  %6.1fs  loss %.6f  policy %.6f  value %.6f  ||  "
              "val_loss %.6f  val_policy %.6f  val_value %.6f" %
              (epoch, time.time() - t0, logs["loss"], logs["policy_loss"],
               logs["value_loss"], logs["val_loss"], logs["val_policy_loss"],
               logs["val_value_loss"]), flush=True)


real_fit = tf.keras.Model.fit


def patched_fit(self, *a, **k):
    k.setdefault("callbacks", [])
    k["callbacks"] = list(k["callbacks"]) + [Timer()]
    k["verbose"] = 0
    return real_fit(self, *a, **k)


tf.keras.Model.fit = patched_fit
try:
    model = TrainMod.train(train_dl, test_dl, model, lr_schedule(0), cfgs,
                           run_dir=os.path.dirname(os.path.abspath(__file__)))
finally:
    tf.keras.Model.fit = real_fit

res = model.evaluate(test_dl, return_dict=True, verbose=0)
print("\nFINAL (best-val checkpoint reloaded):", json.dumps(
    {k: round(float(v), 6) for k, v in res.items()}))
pred = model.predict(te_s, verbose=0)
print("value head: MSE %.6f  corr %.4f  pred std %.4f  target std %.4f" %
      (float(((pred[1][:, 0] - te_v[:, 0]) ** 2).mean()),
       float(np.corrcoef(pred[1][:, 0], te_v[:, 0])[0, 1]),
       pred[1].std(), te_v.std()))
