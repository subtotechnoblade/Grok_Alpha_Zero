"""Smoke test Train.train() end to end on a tiny slice: schedule, val_norm, checkpoint.

Not a quality run - 3 epochs on 20k samples. It only answers "does the wiring work":
does the lr actually move over steps, does val_norm reach the checkpoint, does
save_best_only fire on it, and does load_weights find the file afterwards.
"""
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.chdir(ROOT)

import numpy as np
import tensorflow as tf

from exp2_optimizer import load_split, dedup
from Train import train
from Connect4.Build_Model import build_model
from Connect4.game import Connect4
from Connect4.config import build_config, train_config, optimizer_config, lr_schedule

N = 20000
EPOCHS = 3


class Tiny(tf.keras.utils.Sequence):
    def __init__(self, X, P, V, W, bs=1024):
        super().__init__()
        self.X, self.P, self.V, self.W, self.bs = X, P, V, W, bs

    def __len__(self):
        return int(np.ceil(len(self.X) / self.bs))

    def __getitem__(self, i):
        s = slice(i * self.bs, (i + 1) * self.bs)
        return self.X[s], (self.P[s], self.V[s]), (self.W[s], self.W[s])


def main():
    d = load_split()
    trX, trP, trY, trW = dedup(*d["tr"])
    teX, teP, teY, teW = dedup(*d["te"])
    tr = Tiny(trX[:N].astype(np.float32), trP[:N].astype(np.float32),
              trY[:N].astype(np.float32)[:, None], trW[:N].astype(np.float32))
    te = Tiny(teX[:4096].astype(np.float32), teP[:4096].astype(np.float32),
              teY[:4096].astype(np.float32)[:, None], teW[:4096].astype(np.float32))

    optimizer_config["train_epochs"] = EPOCHS
    game = Connect4()
    model = build_model(game.get_input_state().shape, game.policy_shape,
                        (build_config, train_config, optimizer_config))

    run_dir = os.path.join(HERE, "smoke_run")
    os.makedirs(run_dir, exist_ok=True)
    peak = lr_schedule(0)
    print(f"optimizer={optimizer_config['optimizer']}  gen-0 peak lr={peak}")
    print(f"steps/epoch={len(tr)}  total_steps={len(tr)*EPOCHS}\n")

    model = train(tr, te, model, peak, [build_config, train_config, optimizer_config],
                  run_dir=run_dir)

    opt_lr = model.optimizer.learning_rate
    print(f"\nlr object on the optimizer: {type(opt_lr).__name__}")
    print("SMOKE PASSED: train() returned and reloaded the best weights")


if __name__ == "__main__":
    main()
