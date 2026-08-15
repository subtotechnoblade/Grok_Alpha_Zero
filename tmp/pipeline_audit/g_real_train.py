"""Real data, real config: what val_loss actually measures.

  * splits the val set into the part that is VERBATIM in train ("contaminated")
    and the part that is not ("clean"), and reports val_value_loss on each,
    per epoch.  Only the clean number is a generalisation measure.
  * compares fit's reported TRAIN loss (BatchNorm in training mode, batch stats)
    with evaluate() on the SAME training data (BatchNorm in inference mode,
    moving averages) - the mode that self-play/ONNX actually runs in.
"""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
REPO = r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero"
sys.path.insert(0, REPO)
import numpy as np, h5py as h5, tensorflow as tf
from pathlib import Path

EPOCHS = int(os.environ.get("EPOCHS", 8))
ARCHIVE = os.path.join(REPO, "tmp", "run3_archive_gen0-15_bottleneck_preweightfix")
import Dataloader as DL

sp = DL.Create_Train_Test_Split(ARCHIVE, 4, 1.0, 0.2, 0.1, 0.02, dedup_average=True)
(trs, trp, trv, trw), (tes, tep, tev, tew) = sp.split()
trs_a, tes_a = np.asarray(trs), np.asarray(tes)
trv_a, tev_a = np.asarray(trv), np.asarray(tev)
keys = {trs_a[i].tobytes() for i in range(len(trs_a))}
seen = np.array([tes_a[i].tobytes() in keys for i in range(len(tes_a))])
print("test uniq %d | contaminated %d (%.1f%%, %.1f%% of weight) | clean %d" %
      (len(tes_a), seen.sum(), 100 * seen.mean(),
       100 * tew[seen].sum() / tew.sum(), (~seen).sum()))

from Connect4.game import Connect4
from Connect4.config import build_config, train_config, optimizer_config, lr_schedule
from Connect4.Build_Model import build_model
import Train as TrainMod

game = Connect4()
configs = (build_config, train_config, dict(optimizer_config, train_epochs=EPOCHS))
model = build_model(game.get_input_state().shape, game.policy_shape, configs)
BS = train_config["train_batch_size"]
train_dl = DL.Dataloader(trs, trp, trv, BS, weights=trw)
test_dl = DL.Dataloader(tes, tep, tev, BS, weights=tew)
tr_eval = DL.Dataloader(trs, trp, trv, BS, weights=trw)      # same data, evaluate()
clean_dl = DL.Dataloader([tes[i] for i in np.where(~seen)[0]],
                         [tep[i] for i in np.where(~seen)[0]],
                         [tev[i] for i in np.where(~seen)[0]], BS,
                         weights=tew[~seen] / tew[~seen].mean())
dirty_dl = DL.Dataloader([tes[i] for i in np.where(seen)[0]],
                         [tep[i] for i in np.where(seen)[0]],
                         [tev[i] for i in np.where(seen)[0]], BS,
                         weights=tew[seen] / tew[seen].mean())
print("batches: train %d val %d clean %d dirty %d" %
      (len(train_dl), len(test_dl), len(clean_dl), len(dirty_dl)))


class Split(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        t = self.model.evaluate(tr_eval, return_dict=True, verbose=0)
        c = self.model.evaluate(clean_dl, return_dict=True, verbose=0)
        d = self.model.evaluate(dirty_dl, return_dict=True, verbose=0)
        print("  ep%-3d | TRAIN value: fit(BN=train) %.4f  evaluate(BN=infer) %.4f "
              "| VAL value: all %.4f  contaminated %.4f  CLEAN %.4f "
              "| policy: train %.4f val %.4f" %
              (epoch + 1, logs["value_loss"], t["value_loss"], logs["val_value_loss"],
               d["value_loss"], c["value_loss"], logs["policy_loss"],
               logs["val_policy_loss"]), flush=True)


real_fit = tf.keras.Model.fit


def patched(self, *a, **k):
    k["callbacks"] = list(k.get("callbacks") or []) + [Split()]
    k["verbose"] = 0
    return real_fit(self, *a, **k)


tf.keras.Model.fit = patched
t0 = time.time()
try:
    model = TrainMod.train(train_dl, test_dl, model, lr_schedule(0), configs,
                           run_dir=os.path.dirname(os.path.abspath(__file__)))
finally:
    tf.keras.Model.fit = real_fit
print("elapsed %.0fs" % (time.time() - t0))

pv = model.predict(tes_a, verbose=0)[1][:, 0]
for name, m in [("contaminated", seen), ("clean", ~seen)]:
    w = tew[m]
    print("BEST-CHECKPOINT value head on %-13s: weighted MSE %.4f | pred std %.4f "
          "target std %.4f | corr %.4f | MSE of predicting the train mean %.4f" %
          (name, (w * (pv[m] - tev_a[m, 0]) ** 2).sum() / w.sum(), pv[m].std(),
           tev_a[m, 0].std(), np.corrcoef(pv[m], tev_a[m, 0])[0, 1],
           (w * (tev_a[m, 0] - trv_a[:, 0].mean()) ** 2).sum() / w.sum()))
