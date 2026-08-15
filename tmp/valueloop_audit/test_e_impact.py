"""Impact measurement of the no-op sample weights on the REAL gen-9..14 window
(the data that trained model 15).

1. Rebuild the pooled window, dedup exactly like Dataloader.dedup_average.
2. Compare INTENDED mass (count/N_total, what the weights were supposed to
   restore) vs EFFECTIVE mass (1/N_unique, what broadcast weighting delivers)
   for the most repeated states.
3. Load model 15 and measure how well it fits the averaged value target on the
   most-common opening states vs on unique (count==1) states.
"""
import sys, os
sys.path.insert(0, r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import h5py as h5

ROOT = r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero\Connect4\Grok_Zero_Train"
GENS = [9, 10, 11, 12, 13, 14]  # window that trained model 15

states, policies, values = [], [], []
for g in GENS:
    try:
        f = h5.File(os.path.join(ROOT, str(g), "Self_Play_Data.h5"), mode="r")
    except OSError as e:
        print(f"gen {g}: {e}")
        continue
    n = (len(f.keys()) - 1) // 3
    for i in range(n):
        states.append(f[f"boards_{i}"][:])
        policies.append(f[f"policies_{i}"][:])
        values.append(f[f"values_{i}"][:])
    f.close()

states = np.concatenate(states)
policies = np.concatenate(policies)
values = np.concatenate(values)
N_total = len(states)

flat = states.reshape(N_total, -1)
uniq, inverse, counts = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
N_uniq = len(uniq)
val_means = np.zeros((N_uniq, 1))
np.add.at(val_means, inverse, values.astype(np.float64))
val_means /= counts[:, None]
pol_means = np.zeros((N_uniq, 7))
np.add.at(pol_means, inverse, policies.astype(np.float64))
pol_means /= counts[:, None]

order = np.argsort(counts)[::-1]
print(f"window gens {GENS}: N_total={N_total}  N_unique={N_uniq}  "
      f"duplicates removed: {100 * (1 - N_uniq / N_total):.1f}%")
print(f"count==1 states: {np.sum(counts == 1)} ({100 * np.mean(counts == 1):.1f}% of unique)")
print()
print("top repeated states: intended mass (count/N_total) vs effective mass (1/N_unique)")
for k in order[:5]:
    stones = int(np.abs(uniq[k].reshape(6, 7, 4)[..., 3]).sum())
    intended = counts[k] / N_total
    effective = 1 / N_uniq
    print(f"  count={counts[k]:6d} stones={stones}  intended={100*intended:.3f}%  "
          f"effective={100*effective:.5f}%  underweight x{intended/effective:,.0f}  "
          f"avg value target={val_means[k][0]:+.3f}")

mass_top = counts[order[:100]].sum() / N_total
print(f"\ntop-100 most repeated states: {100*mass_top:.1f}% of intended training mass, "
      f"{100*100/N_uniq:.2f}% effective -> the opening lost "
      f"{mass_top/(100/N_uniq):,.0f}x its weight")

# ---------------------------------------------------------------- model 15 fit
import tensorflow as tf
from Connect4.game import Connect4
from Connect4.Build_Model import build_model
from Connect4.config import build_config, train_config

game = Connect4()
model = build_model(game.get_input_state().shape, game.policy_shape,
                    (build_config, train_config, None))
model.load_weights(os.path.join(ROOT, "15", "model.weights.h5"))

# predict on the 200 most-common states and 200 random unique states
top_idx = order[:200]
rng = np.random.default_rng(1)
uni_idx = rng.choice(np.where(counts == 1)[0], size=200, replace=False)

for name, idx in [("most-common 200 (opening)", top_idx),
                  ("random count==1 200 (mid/endgame)", uni_idx)]:
    x = uniq[idx].reshape(-1, 6, 7, 4).astype(np.float32)
    p_hat, v_hat = model.predict(x, verbose=0)
    v_t = val_means[idx].ravel()
    mse = float(np.mean((v_hat.ravel() - v_t) ** 2))
    # weighted (intended) mse contribution
    print(f"\n{name}:")
    print(f"  value: pred mean {v_hat.mean():+.3f} std {v_hat.std():.3f} | "
          f"target mean {v_t.mean():+.3f} std {v_t.std():.3f} | MSE {mse:.4f}")
    ce = -np.sum(pol_means[idx] * np.log(np.clip(p_hat, 1e-9, 1)), axis=1)
    ce0 = -np.sum(pol_means[idx] * np.log(np.clip(pol_means[idx], 1e-9, 1)), axis=1)
    print(f"  policy: CE {ce.mean():.4f} vs target self-entropy {ce0.mean():.4f} "
          f"(gap {ce.mean()-ce0.mean():.4f})")
    agree = np.mean(np.argmax(p_hat, 1) == np.argmax(pol_means[idx], 1))
    print(f"  policy argmax agreement: {agree:.3f}")

# empty board specifically
x0 = game.get_input_state()[None].astype(np.float32)
k0 = np.where((flat == x0.astype(np.int8).reshape(1, -1)).all(1))[0]
p0, v0 = model.predict(x0, verbose=0)
if len(k0):
    # find its unique row
    u0 = inverse[k0[0]]
    print(f"\nempty board: count={counts[u0]}, avg value target={val_means[u0][0]:+.4f}, "
          f"model value={float(v0[0][0]):+.4f}")
    print(f"  avg policy target={np.round(pol_means[u0], 3).tolist()}")
    print(f"  model policy     ={np.round(p0[0], 3).tolist()}")
