"""What does the training data say the empty-board policy target is,
and what does the gen-9 keras net actually predict?

Gen-9 weights were produced by Train_NN(generation=8): Create_Train_Test_Split
at that moment saw folders 0..8, took files[-5:] = gens 4..8, dropped the oldest
(generation>=2) -> window = gens 5,6,7,8. Gen 9's own Self_Play_Data.h5 is what
the gen-9 onnx then played."""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
REPO = "C:/Users/TryHarder/Desktop/Pythoncode/Grok_Zero"
sys.path.insert(0, REPO)

import numpy as np
import h5py as h5

np.set_printoptions(precision=5, suppress=True)

from Connect4.game import Connect4
game = Connect4()
empty = game.get_input_state()  # (6,7,4) int8, ch0 = 1
empty_bytes = empty.astype(np.int8).tobytes()

def gen_stats(gen):
    path = f"{REPO}/Connect4/Grok_Zero_Train/{gen}/Self_Play_Data.h5"
    pols, vals, first_not_empty = [], [], 0
    with h5.File(path, "r") as f:
        num_datasets = (len(f.keys()) - 1) // 3
        num_games = int(f["game_stats"][2])
        aug = num_datasets // num_games if num_games and num_datasets % num_games == 0 else 1
        for d in range(num_datasets):
            b0 = np.asarray(f[f"boards_{d}"][0], dtype=np.int8)
            if b0.tobytes() != empty_bytes:
                first_not_empty += 1
                continue
            pols.append(np.asarray(f[f"policies_{d}"][0], dtype=np.float64))
            vals.append(float(np.asarray(f[f"values_{d}"][0]).ravel()[0]))
    return num_datasets, num_games, aug, np.array(pols), np.array(vals), first_not_empty

window_pols, window_vals = [], []
for gen in [5, 6, 7, 8, 9]:
    nd, ng, aug, pols, vals, bad = gen_stats(gen)
    mean = pols.mean(0)
    print(f"gen {gen}: {ng} games x {aug} augs = {nd} triples, "
          f"{len(pols)} empty-board samples ({bad} first-rows not empty)")
    print(f"   mean empty-board policy target: {mean}  argmax={mean.argmax()}")
    print(f"   mean value target: {vals.mean():+.4f}")
    if gen <= 8:
        window_pols.append(pols); window_vals.append(vals)

pooled = np.concatenate(window_pols, 0)
pooled_mean = pooled.mean(0)
print("\n---- pooled training window (gens 5-8), incl. mirror augmentation ----")
print("dedup'd empty-board policy TARGET:", pooled_mean, " argmax:", pooled_mean.argmax())
print("dedup'd empty-board value  TARGET:", np.concatenate(window_vals).mean())
print("symmetric check (should be ~0):", np.abs(pooled_mean - pooled_mean[::-1]).max())

# per-column vote shares: fraction of games whose target argmax is each column
am = pooled.argmax(1)
print("argmax distribution over window samples:", np.bincount(am, minlength=7) / len(am))

# ---- the net ----
import tensorflow as tf
from Connect4.config import build_config, train_config, optimizer_config
from Connect4.Build_Model import build_model
configs = (build_config, train_config, optimizer_config)
model = build_model(game.get_input_state().shape, game.policy_shape, configs)
model.load_weights(f"{REPO}/Connect4/Grok_Zero_Train/9/model.weights.h5")
kp, kv = model.predict(empty.astype(np.float32)[None], verbose=0)
kp = np.asarray(kp)[0]; kv = float(np.asarray(kv).ravel()[0])
print("\n---- gen-9 keras net on empty board ----")
print("net policy:", kp, " argmax:", kp.argmax())
print("net value :", kv)
ce_target = -(pooled_mean * np.log(np.clip(kp, 1e-12, 1))).sum()
ce_self = -(pooled_mean * np.log(np.clip(pooled_mean, 1e-12, 1))).sum()
print(f"CE(target, net) = {ce_target:.4f}, entropy(target) = {ce_self:.4f}, "
      f"KL = {ce_target - ce_self:.4f}")

# also: gen-8 net (the weights training STARTED from) for reference
model.load_weights(f"{REPO}/Connect4/Grok_Zero_Train/8/model.weights.h5")
kp8, kv8 = model.predict(empty.astype(np.float32)[None], verbose=0)
kp8 = np.asarray(kp8)[0]
print("\ngen-8 net policy on empty board:", kp8, " argmax:", kp8.argmax())
