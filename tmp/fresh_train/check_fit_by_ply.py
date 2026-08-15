"""How well does the gen-9 net fit its own (dedup'd) training window, by ply?
Pools gens 5-8, dedups exactly like Dataloader.dedup_average, runs the net,
reports KL(target || net) bucketed by ply. Also counts unique states."""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
REPO = "C:/Users/TryHarder/Desktop/Pythoncode/Grok_Zero"
sys.path.insert(0, REPO)

import numpy as np
import h5py as h5
np.set_printoptions(precision=5, suppress=True)

states, policies, values = [], [], []
for gen in [5, 6, 7, 8]:
    with h5.File(f"{REPO}/Connect4/Grok_Zero_Train/{gen}/Self_Play_Data.h5", "r") as f:
        nd = (len(f.keys()) - 1) // 3
        for d in range(nd):
            states.append(np.asarray(f[f"boards_{d}"][:], dtype=np.int8))
            policies.append(np.asarray(f[f"policies_{d}"][:], dtype=np.float64))
            values.append(np.asarray(f[f"values_{d}"][:], dtype=np.float64))
states = np.concatenate(states, 0)
policies = np.concatenate(policies, 0)
values = np.concatenate(values, 0).reshape(len(states), -1)
print("raw samples:", len(states))

flat = states.reshape(len(states), -1)
uniq, inverse, counts = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
pol_mean = np.zeros((len(uniq), policies.shape[1]))
val_mean = np.zeros((len(uniq), values.shape[1]))
np.add.at(pol_mean, inverse, policies)
np.add.at(val_mean, inverse, values)
pol_mean /= counts[:, None]
val_mean /= counts[:, None]
uniq_states = uniq.reshape(-1, *states.shape[1:])
print("unique samples after dedup:", len(uniq_states),
      f"({100 * (1 - len(uniq_states) / len(states)):.1f}% removed)")

# ply = stones on the board (channel 3 = current board)
ply = np.count_nonzero(uniq_states[:, :, :, 3].reshape(len(uniq_states), -1), axis=1)

import tensorflow as tf
from Connect4.game import Connect4
from Connect4.config import build_config, train_config, optimizer_config
from Connect4.Build_Model import build_model
game = Connect4()
configs = (build_config, train_config, optimizer_config)
model = build_model(game.get_input_state().shape, game.policy_shape, configs)
model.load_weights(f"{REPO}/Connect4/Grok_Zero_Train/9/model.weights.h5")

pred_p, pred_v = model.predict(uniq_states.astype(np.float32), batch_size=2048, verbose=0)
pred_p = np.clip(np.asarray(pred_p, dtype=np.float64), 1e-12, 1.0)
pred_v = np.asarray(pred_v, dtype=np.float64).reshape(len(uniq_states), -1)

t = np.clip(pol_mean, 1e-12, 1.0)
kl = (pol_mean * (np.log(t) - np.log(pred_p))).sum(1)
vmse = ((val_mean - pred_v) ** 2).mean(1)
agree = (pol_mean.argmax(1) == pred_p.argmax(1))

print(f"\noverall: mean KL={kl.mean():.4f}  median KL={np.median(kl):.4f}  "
      f"argmax agree={agree.mean():.3f}  value MSE={vmse.mean():.4f}")

buckets = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 6), (7, 12), (13, 20), (21, 42)]
print(f"\n{'ply':>7} {'n_uniq':>7} {'occ_mass%':>9} {'meanKL':>8} {'argmax_agree':>12} {'valMSE':>8}")
occ_total = counts.sum()
for lo, hi in buckets:
    m = (ply >= lo) & (ply <= hi)
    if m.sum() == 0: continue
    mass = counts[m].sum() / occ_total * 100
    print(f"{lo:>3}-{hi:<3} {m.sum():>7} {mass:>9.2f} {kl[m].mean():>8.4f} "
          f"{agree[m].mean():>12.3f} {vmse[m].mean():>8.4f}")

i0 = np.where(ply == 0)[0]
print("\nempty board: count in raw window =", counts[i0[0]] if len(i0) else 0,
      "of", occ_total, f"-> weight before dedup {counts[i0[0]]/occ_total*100:.2f}%, after {1/len(uniq_states)*100:.4f}%")
print("empty-board target:", pol_mean[i0[0]])
print("empty-board net   :", pred_p[i0[0]])
