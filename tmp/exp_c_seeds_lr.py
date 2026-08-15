"""Experiment C: experiment B across 3 seeds, plus a learning rate sweep.

Two questions:
  1. is the buggy-vs-fixed difference bigger than seed noise?
  2. the fix makes trunk steps ~6x larger, so what lr should the fixed version use?

The `buggy lr 6e-2` arm is a control: it gives the buggy reshape roughly the same
trunk step SIZE as fixed@1e-2, which separates "wrong step size" from "no whitening".

    python tmp/exp_c_seeds_lr.py
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

EPOCHS, BATCH = 6, 256
MAX_TRAIN, MAX_TEST = 6144, 2048
SEEDS = [7, 13, 29]
CONFIGS = [("buggy 1e-2", BuggyMuon, 1e-2),
           ("buggy 6e-2 (step matched)", BuggyMuon, 6e-2),
           ("fixed 1e-2", Muon, 1e-2),
           ("fixed 2e-3", Muon, 2e-3),
           ("fixed 1e-3", Muon, 1e-3)]


def get_data(seed):
    (trs, trp, trv), (tes, tep, tev) = Create_Train_Test_Split(
        os.path.join(ROOT, "Connect4", "Grok_Zero_Train"), num_previous_generations=1,
        train_percent=1.0, train_decay=0.2,
        test_percent=0.25, test_decay=0.03).split()
    rng = np.random.RandomState(seed)
    tr_i, te_i = rng.permutation(len(trs))[:MAX_TRAIN], rng.permutation(len(tes))[:MAX_TEST]
    return (Dataloader([trs[i] for i in tr_i], [trp[i] for i in tr_i], [trv[i] for i in tr_i], BATCH),
            Dataloader([tes[i] for i in te_i], [tep[i] for i in te_i], [tev[i] for i in te_i], BATCH))


def run(cls, lr, seed, train, test):
    tf.keras.utils.set_random_seed(seed)
    game = Connect4()
    model = build_model(game.get_input_state().shape, game.policy_shape,
                        (build_config, train_config, optimizer_config))
    kwargs = dict(optimizer_config["kwargs"]); kwargs["learning_rate"] = lr
    opt = optimizer_cls_wrap(cls(**kwargs))
    model.compile(optimizer=opt,
                  loss={"policy": Policy_Loss(dtype="float32"), "value": Value_Loss(dtype="float32")},
                  metrics={"policy": KLD(dtype="float32")})
    h = model.fit(train, validation_data=test, epochs=EPOCHS, verbose=0)
    return min(h.history["val_loss"]), h.history["loss"][-1], h.history


def optimizer_cls_wrap(opt):
    return Orthograd(opt) if optimizer_config.get("use_orthograd", False) else opt


def main():
    t0 = time.time()
    out = {}
    for seed in SEEDS:
        train, test = get_data(seed)
        for label, cls, lr in CONFIGS:
            bv, tl, hist = run(cls, lr, seed, train, test)
            out.setdefault(label, {"best_val": [], "final_train": [], "hist": []})
            out[label]["best_val"].append(bv)
            out[label]["final_train"].append(tl)
            out[label]["hist"].append({k: [float(x) for x in v] for k, v in hist.items()})
            print(f"  seed {seed}  {label:<28} best val {bv:.4f}  final train {tl:.4f}", flush=True)

    print(f"\n{'config':<28} {'best val loss':>22} {'final train loss':>22}")
    print("-" * 74)
    for label, _, _ in CONFIGS:
        bv, tl = np.array(out[label]["best_val"]), np.array(out[label]["final_train"])
        print(f"{label:<28} {bv.mean():>10.4f} +/- {bv.std():<8.4f} {tl.mean():>10.4f} +/- {tl.std():<8.4f}")
    print("-" * 74)
    print(f"total {time.time() - t0:.0f}s over {len(SEEDS)} seeds")

    with open(os.path.join(ROOT, "tmp", "results_seeds_lr.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote tmp/results_seeds_lr.json")


if __name__ == "__main__":
    main()
