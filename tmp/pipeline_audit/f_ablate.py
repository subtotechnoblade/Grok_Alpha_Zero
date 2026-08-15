"""ITEM 6b: ablation on the noiseless synthetic value target.

Same data / model / loss / dataloader as e_synthetic.py. Only the optimizer stack
varies, so whatever moves the value head's output SCALE is the culprit.
"""
import os, sys, time, json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
REPO = r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero"
sys.path.insert(0, REPO)
import numpy as np, h5py as h5, tensorflow as tf

EPOCHS = int(os.environ.get("EPOCHS", 25))
N_TRAIN, N_TEST = 12000, 3000
ARCHIVE = os.path.join(REPO, "tmp", "run3_archive_gen0-15_bottleneck_preweightfix")

states = []
for gen in ["15", "14", "13", "12", "11", "10", "9", "8"]:
    try:
        with h5.File(os.path.join(ARCHIVE, gen, "Self_Play_Data.h5"), "r") as f:
            nd = (len(f.keys()) - 1) // 3
            for ds in range(nd):
                states.append(f[f"boards_{ds}"][:])
    except OSError as e:
        print("h5 fail", gen, e)
    n = len(np.unique(np.concatenate(states, 0).reshape(-1, 168), axis=0))
    if n > 1.3 * (N_TRAIN + N_TEST):
        break
S = np.concatenate(states, 0)
S = np.unique(S.reshape(len(S), -1), axis=0).reshape(-1, 6, 7, 4).astype(np.int8)
np.random.RandomState(0).shuffle(S)
S = S[:N_TRAIN + N_TEST]
board = S[:, :, :, 3]
V = ((board == 1).sum((1, 2)).astype(np.float32) / 10.5 - 1.0).reshape(-1, 1)
legal = ((board != 0).sum(1) < 6).astype(np.float32)
P = (legal / legal.sum(1, keepdims=True)).astype(np.float32)
tr = slice(0, N_TRAIN); te = slice(N_TRAIN, None)
rng = np.random.RandomState(1)
c = rng.pareto(1.2, N_TRAIN) + 1.0; trw = (c * (N_TRAIN / c.sum())).astype(np.float32)
c2 = rng.pareto(1.2, len(S) - N_TRAIN) + 1.0
tew = (c2 * (len(c2) / c2.sum())).astype(np.float32)
print("N=%d  value target std %.4f  var %.4f" % (len(S), V[te].std(), V[te].var()))

from Connect4.game import Connect4
from Connect4.config import build_config, train_config, optimizer_config, lr_schedule
from Connect4.Build_Model import build_model
from Dataloader import Dataloader
from Net.Alpha_Loss import Policy_Loss, Value_Loss
from Net.Optimizers.muon import Muon
from Net.Optimizers.adam import Adam
from Net.Optimizers.ortho_grad import Orthograd

game = Connect4()
configs = (build_config, train_config, optimizer_config)
train_dl = Dataloader(list(S[tr]), list(P[tr]), list(V[tr]), 1024, weights=trw)
test_dl = Dataloader(list(S[te]), list(P[te]), list(V[te]), 1024, weights=tew)
LR = lr_schedule(0)


def make_opt(kind):
    kw = dict(optimizer_config["kwargs"]); kw["learning_rate"] = LR
    if kind.startswith("muon"):
        o = Muon(**kw)
    else:
        o = Adam(learning_rate=LR)
    if kind.endswith("+ortho"):
        o = Orthograd(o)
    return o


class Probe(tf.keras.callbacks.Callback):
    def __init__(self, tag):
        self.tag = tag; self.rows = []
    def on_epoch_end(self, epoch, logs=None):
        k = [v for v in self.model.trainable_variables if "value_1" in v.path and "kernel" in v.path][0]
        b = [v for v in self.model.trainable_variables if "value_1" in v.path and "bias" in v.path][0]
        pk = [v for v in self.model.trainable_variables if "policy_1" in v.path and "kernel" in v.path][0]
        pred = self.model.predict(S[te], verbose=0)[1][:, 0]
        r = float(np.corrcoef(pred, V[te][:, 0])[0, 1])
        mse = float(((pred - V[te][:, 0]) ** 2).mean())
        best = float(V[te].var() * (1 - r ** 2))   # MSE if the scale were optimal
        self.rows.append((epoch, logs["value_loss"], logs["val_value_loss"],
                          float(np.linalg.norm(k.numpy())), float(b.numpy()[0]),
                          float(np.linalg.norm(pk.numpy())), pred.std(), r, mse, best))
        if epoch % 4 == 0 or epoch == EPOCHS - 1:
            print("   %-13s ep%-3d tr_val %.4f val_val %.4f ||value_1 W|| %.5f b %+.4f "
                  "||policy_1 W|| %.4f pred_std %.4f r %.4f mse %.4f (best-scaled %.4f)"
                  % (self.tag, epoch, logs["value_loss"], logs["val_value_loss"],
                     np.linalg.norm(k.numpy()), b.numpy()[0], np.linalg.norm(pk.numpy()),
                     pred.std(), r, mse, best), flush=True)


results = {}
for kind in ["muon+ortho", "muon", "adam+ortho", "adam"]:
    tf.keras.utils.set_random_seed(0)
    m = build_model(game.get_input_state().shape, game.policy_shape, configs)
    m.compile(optimizer=make_opt(kind),
              loss={"policy": Policy_Loss(dtype="float32"),
                    "value": Value_Loss(dtype="float32")},
              weighted_metrics={"policy": tf.keras.metrics.KLDivergence(name="policy_kld")})
    print("\n=== %s (lr %.1e, %d epochs) ===" % (kind, LR, EPOCHS), flush=True)
    pr = Probe(kind)
    t0 = time.time()
    m.fit(train_dl, validation_data=test_dl, epochs=EPOCHS, verbose=0, callbacks=[pr])
    results[kind] = pr.rows
    last = pr.rows[-1]
    print("   -> %.0fs | final val_value %.4f | pred_std %.4f vs target %.4f | "
          "r %.4f | mse %.4f | mse if scale were right %.4f"
          % (time.time() - t0, last[2], last[6], V[te].std(), last[7], last[8], last[9]))

json.dump({k: [list(map(float, r)) for r in v] for k, v in results.items()},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "f_ablate.json"), "w"))
print("\n%-12s %-10s %-10s %-9s %-8s %-8s" % ("opt", "val_value", "pred_std", "corr", "mse", "||Wv||"))
for k, v in results.items():
    l = v[-1]
    print("%-12s %-10.4f %-10.4f %-9.4f %-8.4f %-8.5f" % (k, l[2], l[6], l[7], l[8], l[3]))
print("target std %.4f  var %.4f" % (V[te].std(), V[te].var()))
