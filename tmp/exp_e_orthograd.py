"""Experiment E: does Orthograd earn its place?

Every run in experiments B-D had use_orthograd=True, so OrthoGrad was never the
variable. This is the full 2x2: {buggy, fixed} reshape x {with, without} Orthograd,
swept over the learning rate basin found in experiment D, 3 seeds each.

Also records trunk weight norm, since pinning ||W|| is OrthoGrad's whole mechanism.

    python tmp/exp_e_orthograd.py
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
LRS = [1e-3, 3e-3]
# Minimal grid: experiments C and D already cover Orthograd(Muon) at 1e-3/2e-3/5e-3/1e-2
# and Orthograd(BuggyMuon) at 1e-3/3e-3/1e-2, all over these same 3 seeds. The only arm
# with no data at all is Muon WITHOUT Orthograd, so that is all this runs. 6 runs.
ARMS = [("fixed, no ortho", Muon, False)]


def get_data(seed):
    (trs, trp, trv), (tes, tep, tev) = Create_Train_Test_Split(
        os.path.join(ROOT, "Connect4", "Grok_Zero_Train"), num_previous_generations=1,
        train_percent=1.0, train_decay=0.2,
        test_percent=0.25, test_decay=0.03).split()
    rng = np.random.RandomState(seed)
    tr_i, te_i = rng.permutation(len(trs))[:MAX_TRAIN], rng.permutation(len(tes))[:MAX_TEST]
    return (Dataloader([trs[i] for i in tr_i], [trp[i] for i in tr_i], [trv[i] for i in tr_i], BATCH),
            Dataloader([tes[i] for i in te_i], [tep[i] for i in te_i], [tev[i] for i in te_i], BATCH))


def trunk_norm(model):
    trunk = [v for v in model.trainable_variables if len(v.shape) == 4 and v.shape[-1] == 128]
    return float(np.mean([float(tf.norm(v)) for v in trunk]))


def main():
    t0, out = time.time(), {}
    for seed in SEEDS:
        train, test = get_data(seed)
        for label, cls, use_ortho in ARMS:
            for lr in LRS:
                tf.keras.utils.set_random_seed(seed)
                game = Connect4()
                model = build_model(game.get_input_state().shape, game.policy_shape,
                                    (build_config, train_config, optimizer_config))
                w0 = trunk_norm(model)
                kwargs = dict(optimizer_config["kwargs"]); kwargs["learning_rate"] = lr
                opt = cls(**kwargs)
                if use_ortho:
                    opt = Orthograd(opt)
                model.compile(optimizer=opt,
                              loss={"policy": Policy_Loss(dtype="float32"),
                                    "value": Value_Loss(dtype="float32")},
                              metrics={"policy": KLD(dtype="float32")})
                h = model.fit(train, validation_data=test, epochs=EPOCHS, verbose=0)
                key = f"{label} @ {lr:g}"
                out.setdefault(key, {"best_val": [], "final_train": [], "wgrowth": []})
                out[key]["best_val"].append(float(min(h.history["val_loss"])))
                out[key]["final_train"].append(float(h.history["loss"][-1]))
                out[key]["wgrowth"].append(trunk_norm(model) / w0)
                print(f"  seed {seed}  {key:<24} best val {min(h.history['val_loss']):.4f}  "
                      f"trunk |W| x{out[key]['wgrowth'][-1]:.3f}", flush=True)

    print(f"\n{'arm':<20}" + "".join(f"{f'lr {lr:g}':>26}" for lr in LRS))
    print(f"{'':<20}" + "".join(f"{'best val    |W| growth':>26}" for _ in LRS))
    print("-" * (20 + 26 * len(LRS)))
    for label, _, _ in ARMS:
        row = f"{label:<20}"
        for lr in LRS:
            d = out[f"{label} @ {lr:g}"]
            bv, wg = np.array(d["best_val"]), np.array(d["wgrowth"])
            row += f"{bv.mean():>10.4f}+/-{bv.std():<5.3f}{wg.mean():>7.2f}x"
        print(row)
    print("-" * (20 + 26 * len(LRS)))
    print(f"total {time.time() - t0:.0f}s")

    with open(os.path.join(ROOT, "tmp", "results_orthograd.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote tmp/results_orthograd.json")


if __name__ == "__main__":
    main()
