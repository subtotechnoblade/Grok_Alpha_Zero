"""How expensive is one game's h5 append, native NTFS vs the layout production uses?

Production (Self_Play.save_data): under a GLOBAL mp.Lock, open the file in append
mode, create 3 datasets per augmentation (aug=2 for Connect4), bump game_stats,
close. One such transaction per finished game, serialized across ALL workers.

If a transaction costs T seconds, aggregate throughput is capped at 1/T games/s
regardless of worker count.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import h5py as h5

PATH = "tmp/bench_h5_file.h5"
GAMES = 200
PLIES = 35
AUG = 2

if os.path.exists(PATH):
    os.remove(PATH)

with h5.File(PATH, "w") as f:
    f.create_dataset("game_stats", data=np.zeros(6, dtype=np.int64))

boards = np.random.rand(PLIES, 6, 7, 4).astype(np.float32)
policies = np.random.rand(PLIES, 7).astype(np.float32)
values = np.random.rand(PLIES, 1).astype(np.float32)

times = []
for g in range(GAMES):
    t0 = time.perf_counter()
    with h5.File(PATH, "a") as f:          # open+close per game, like production
        for a in range(AUG):
            i = g * AUG + a
            f.create_dataset(f"boards_{i}", data=boards, chunks=None)
            f.create_dataset(f"policies_{i}", data=policies, chunks=None)
            f.create_dataset(f"values_{i}", data=values, chunks=None)
        s = f["game_stats"][:]
        s[1] += PLIES
        s[2] += 1
        f["game_stats"][:] = s
    times.append(time.perf_counter() - t0)

times = np.array(times) * 1000
size = os.path.getsize(PATH) / 1e6
print(f"{GAMES} game-transactions, file grew to {size:.1f} MB")
print(f"per game-write: mean {times.mean():6.1f} ms   p50 {np.percentile(times,50):6.1f}"
      f"   p95 {np.percentile(times,95):6.1f}   max {times.max():6.1f}")
print(f"early (first 20) mean: {times[:20].mean():6.1f} ms")
print(f"late (last 20) mean  : {times[-20:].mean():6.1f} ms   <- growth = O(n) reopen cost")
print(f"\nlock-imposed ceiling if writes serialize: {1000/times[-20:].mean():.1f} games/s")
os.remove(PATH)
