"""Reproduce the train->deployment coverage deficit on the OWNER'S OWN archived runs,
so the PROBLEM statement rests on his data, not on my re-generated games.

For generation G: what fraction of the positions gen G's self play actually visited
(weighted by occurrence, by ply band) already appear anywhere in the training window
that produced the gen-G network = generations [G-num_prev, G-1]?

The h5 stores boards_<2k> = game k and boards_<2k+1> = its left-right mirror; only the
even ones are used so a position is not credited to its own mirror. The key is the raw
4-plane input state, i.e. exactly what Dataloader.dedup_average merges on.

usage: archive_coverage.py <archive_dir> <gen> [num_prev]
"""
import sys
import hashlib
from pathlib import Path

import h5py
import numpy as np

BANDS = [(0, 5), (6, 9), (10, 13), (14, 17), (18, 99)]
NAMES = ["0-5", "6-9", "10-13", "14-17", "18+"]


def band_of(p):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= p <= hi:
            return i
    return len(BANDS) - 1


def gen_positions(h5path):
    """-> (keys array, ply array, n_games)"""
    keys, plies = [], []
    n = 0
    with h5py.File(h5path, "r") as f:
        names = [k for k in f.keys() if k.startswith("boards_")]
        ids = sorted(int(k.split("_")[1]) for k in names)
        for i in ids:
            if i % 2:      # odd = mirrored copy of game i//2
                continue
            arr = f[f"boards_{i}"][:]
            n += 1
            for p in range(arr.shape[0]):
                keys.append(hashlib.blake2b(arr[p].tobytes(), digest_size=8).digest())
                plies.append(p)
    return np.array(keys, dtype="S8"), np.array(plies), n


def main():
    arch = Path(sys.argv[1])
    G = int(sys.argv[2])
    num_prev = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    tgt_k, tgt_p, tgt_n = gen_positions(arch / str(G) / "Self_Play_Data.h5")

    # build the window one generation at a time, newest first, so coverage can be read
    # off at 1, 2, ... num_prev generations of history = a MEASURED scaling curve
    windows = []          # (label, set, n_games)
    acc = set()
    acc_games = 0
    for k_prev in range(1, num_prev + 1):
        g = G - k_prev
        p = arch / str(g) / "Self_Play_Data.h5"
        if not p.exists():
            continue
        kk, _, n = gen_positions(p)
        acc.update(kk.tolist())
        acc_games += n
        windows.append((f"gens {G - k_prev}-{G - 1}", set(acc), acc_games))

    print(f"archive {arch.name}")
    print(f"target generation {G}: {tgt_n} games, {len(tgt_k)} positions")
    for lab, s, n in windows:
        print(f"  window {lab:>12}: {n:>5} games, {len(s):>7} distinct positions")
    print()
    print("occurrence-weighted coverage of gen-G positions, by training-window size")
    print(f"{'ply band':>9} {'gen-G pos':>10} {'distinct':>9} " +
          " ".join(f"{str(n) + 'g':>9}" for _, _, n in windows))
    for bi, name in enumerate(NAMES):
        m = np.array([band_of(p) == bi for p in tgt_p])
        kk = tgt_k[m]
        if len(kk) == 0:
            continue
        u = np.unique(kk)
        cells = []
        for _, s, _ in windows:
            hit = np.array([x in s for x in kk.tolist()])
            cells.append(f"{100 * hit.mean():>8.1f}%")
        print(f"{name:>9} {len(kk):>10} {len(u):>9} " + " ".join(cells))


if __name__ == "__main__":
    main()
