"""Experiment B: does the reshape bug actually cost you loss?

Trains the real Connect4 model on real self play data, identical init and identical
data for every config, changing only the optimizer.

    python tmp/exp_b_training.py [epochs]
"""
import os, sys, json, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["NUMBA_DISABLE_JIT"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import tensorflow as tf

from Connect4.Connect4 import Connect4, build_config, train_config, optimizer_config
from Connect4.Build_Model import build_model
from Net.Optimizers.muon import Muon
from Net.Optimizers.ortho_grad import Orthograd
from Net.Alpha_Loss import Policy_Loss, Value_Loss, KLD
from Dataloader import Dataloader, Create_Train_Test_Split
from tmp.muon_buggy import BuggyMuon

SEED = 7
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
BATCH = 256
MAX_TRAIN, MAX_TEST = 6144, 2048


def get_data():
    (trs, trp, trv), (tes, tep, tev) = Create_Train_Test_Split(
        os.path.join(ROOT, "Connect4", "Grok_Zero_Train"), num_previous_generations=1,
        train_percent=1.0, train_decay=0.2,
        test_percent=0.25, test_decay=0.03).split()
    rng = np.random.RandomState(SEED)
    tr_i = rng.permutation(len(trs))[:MAX_TRAIN]
    te_i = rng.permutation(len(tes))[:MAX_TEST]
    train = Dataloader([trs[i] for i in tr_i], [trp[i] for i in tr_i], [trv[i] for i in tr_i], BATCH)
    test = Dataloader([tes[i] for i in te_i], [tep[i] for i in te_i], [tev[i] for i in te_i], BATCH)
    print(f"train {len(tr_i)} positions ({len(train)} steps/epoch)   test {len(te_i)}\n")
    return train, test


def run(label, optimizer_cls, lr, train, test):
    tf.keras.utils.set_random_seed(SEED)          # identical init for every config
    game = Connect4()
    configs = (build_config, train_config, optimizer_config)
    model = build_model(game.get_input_state().shape, game.policy_shape, configs)

    kwargs = dict(optimizer_config["kwargs"]); kwargs["learning_rate"] = lr
    opt = optimizer_cls(**kwargs)
    if optimizer_config.get("use_orthograd", False):
        opt = Orthograd(opt)

    model.compile(optimizer=opt,
                  loss={"policy": Policy_Loss(dtype="float32"), "value": Value_Loss(dtype="float32")},
                  metrics={"policy": KLD(dtype="float32")})
    t0 = time.time()
    h = model.fit(train, validation_data=test, epochs=EPOCHS, verbose=0)
    took = time.time() - t0

    trunk = [v for v in model.trainable_variables if len(v.shape) == 4 and v.shape[-1] == 128]
    norm = float(np.mean([float(tf.norm(v)) for v in trunk]))
    print(f"{label:<26} {took/EPOCHS:>6.1f}s/ep  "
          f"train {h.history['loss'][0]:.4f} -> {h.history['loss'][-1]:.4f}   "
          f"val {h.history['val_loss'][0]:.4f} -> {h.history['val_loss'][-1]:.4f}   "
          f"best val {min(h.history['val_loss']):.4f}   mean trunk ||W|| {norm:.2f}")
    return {k: [float(x) for x in v] for k, v in h.history.items()}


def main():
    train, test = get_data()
    results = {}
    print(f"{'config':<26} {'time':>8}  {'train loss':<24} {'val loss':<24} {'':<16}")
    print("-" * 118)
    results["buggy_lr1e-2"] = run("BUGGY reshape, lr 1e-2", BuggyMuon, 1e-2, train, test)
    results["fixed_lr1e-2"] = run("FIXED reshape, lr 1e-2", Muon, 1e-2, train, test)
    results["fixed_lr2e-3"] = run("FIXED reshape, lr 2e-3", Muon, 2e-3, train, test)
    print("-" * 118)

    print(f"\nper epoch val_loss")
    print(f"{'epoch':>6} " + "".join(f"{k:>18}" for k in results))
    for e in range(EPOCHS):
        print(f"{e:>6} " + "".join(f"{results[k]['val_loss'][e]:>18.4f}" for k in results))
    print(f"\nper epoch policy_loss (train)")
    print(f"{'epoch':>6} " + "".join(f"{k:>18}" for k in results))
    for e in range(EPOCHS):
        print(f"{e:>6} " + "".join(f"{results[k]['policy_loss'][e]:>18.4f}" for k in results))

    with open(os.path.join(ROOT, "tmp", "results_muon_shape.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote tmp/results_muon_shape.json")


if __name__ == "__main__":
    main()
