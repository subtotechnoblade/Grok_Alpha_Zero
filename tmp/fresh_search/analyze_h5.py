"""Analyze self-play data diversity per generation.

For each generation folder, reconstruct each original (non-mirrored) game's move
sequence from consecutive board planes, then report:
- first-move histogram
- distinct game prefixes at plies 1,2,4,6,8,10 and distinct full games
- mean ply-0 policy target (what the net is trained toward at the empty board)
- mean ply-0 value target
"""
import sys
import numpy as np
import h5py as h5
from collections import Counter

def game_moves(boards):
    # boards: (T, 6, 7, 4) int8, plane 3 = current board (pre-move state per ply)
    T = boards.shape[0]
    moves = []
    for t in range(T - 1):
        diff = boards[t + 1][:, :, 3].astype(np.int16) - boards[t][:, :, 3].astype(np.int16)
        ys, xs = np.nonzero(diff)
        if len(xs) != 1:
            return moves, False
        moves.append(int(xs[0]))
    return moves, True

def analyze(folder):
    path = f"{folder}/Self_Play_Data.h5"
    with h5.File(path, "r") as f:
        stats = f["game_stats"][:]
        n_datasets = (len(f.keys()) - 1) // 3
        first_moves = Counter()
        prefixes = {k: set() for k in (1, 2, 4, 6, 8, 10)}
        full_games = set()
        ply0_policies = []
        ply0_values = []
        n_games = 0
        for k in range(0, n_datasets, 2):  # even = original orientation
            b = f[f"boards_{k}"][:]
            moves, ok = game_moves(b)
            if not ok or len(moves) == 0:
                continue
            n_games += 1
            first_moves[moves[0]] += 1
            for p in prefixes:
                if len(moves) >= p:
                    prefixes[p].add(tuple(moves[:p]))
            full_games.add(tuple(moves))
            ply0_policies.append(f[f"policies_{k}"][0])
            ply0_values.append(f[f"values_{k}"][0])
        mp = np.mean(ply0_policies, axis=0) if ply0_policies else None
        mv = float(np.mean(ply0_values)) if ply0_values else None
    print(f"=== {folder} ===")
    print(f"game_stats [max_len, total_actions, games, p-1 wins, draws, p+1 wins] = {stats}")
    print(f"games reconstructed: {n_games}")
    print(f"first-move histogram (col: count): {dict(sorted(first_moves.items()))}")
    for p in sorted(prefixes):
        print(f"  distinct prefixes at ply {p:2d}: {len(prefixes[p])}")
    print(f"  distinct full games: {len(full_games)}")
    if mp is not None:
        print(f"  mean ply-0 policy target: {np.array2string(mp, precision=3, suppress_small=True)}")
        print(f"  mean ply-0 value target : {mv:+.3f}")
    print()

if __name__ == "__main__":
    gens = sys.argv[1:] or ["9"]
    for g in gens:
        analyze(f"Connect4/Grok_Zero_Train/{g}")
