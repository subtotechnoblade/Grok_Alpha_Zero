"""Ply-0 policy/value tables per generation, from RAW h5 (before dedup).

For each generation:
  - mean policy target per column over ALL stored ply-0 samples (both augmentation copies)
  - mean policy target per column over ORIGINAL copies only (k=0) -> the true search output
  - mean value target at ply 0 (mover = player -1)
  - histogram of the actually-played first move (derived from board diff ply0 -> ply1)
"""
import h5py as h5
import numpy as np

GENS = range(10)
BASE = "Connect4/Grok_Zero_Train"

for gen in GENS:
    p = f"{BASE}/{gen}/Self_Play_Data.h5"
    with h5.File(p, "r") as f:
        num_trip = (len(f.keys()) - 1) // 3
        num_games = int(f["game_stats"][2])
        augs = num_trip // num_games

        pol_all, pol_orig, val0, first_moves, lengths = [], [], [], [], []
        for g in range(num_games):
            for k in range(augs):
                d = g * augs + k
                pol = f[f"policies_{d}"][0]
                pol_all.append(pol)
                if k == 0:
                    pol_orig.append(pol)
                    val0.append(f[f"values_{d}"][0][0])
                    boards = f[f"boards_{d}"]
                    lengths.append(boards.shape[0])
                    if boards.shape[0] >= 2:
                        b0 = boards[0][:, :, 3].astype(np.int16)
                        b1 = boards[1][:, :, 3].astype(np.int16)
                        diff = np.argwhere(b1 - b0 != 0)
                        assert len(diff) == 1
                        first_moves.append(int(diff[0][1]))

        pol_all = np.array(pol_all)
        pol_orig = np.array(pol_orig)
        fm = np.bincount(np.array(first_moves), minlength=7)
        stats = f["game_stats"][:]
        print(f"gen {gen}: games={num_games} p-1w/draw/p1w = "
              f"{stats[3]/num_games:.3f}/{stats[4]/num_games:.3f}/{stats[5]/num_games:.3f} "
              f"avg_len={np.mean(lengths):.1f}")
        with np.printoptions(precision=3, suppress=True):
            print("  mean ply0 policy (stored, both copies):", pol_all.mean(0))
            print("  mean ply0 policy (original copy only): ", pol_orig.mean(0))
            print("  first move actually played, counts:    ", fm, f" argmax_col={fm.argmax()}")
            print(f"  mean ply0 value (mover=-1): {np.mean(val0):+.4f}   "
                  f"top col of orig policy: {pol_orig.mean(0).argmax()}")
