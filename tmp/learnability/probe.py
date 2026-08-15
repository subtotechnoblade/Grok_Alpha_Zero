"""Probe the static archives: shapes, dtypes, aug factor, value decomposition."""
import sys
import numpy as np
import h5py as h5

RUN2 = "C:/Users/TryHarder/Desktop/Pythoncode/Grok_Zero/tmp/run2_archive_gen0-19_oldvaluehead"
RUN3 = "C:/Users/TryHarder/Desktop/Pythoncode/Grok_Zero/tmp/run3_archive_gen0-15_bottleneck_preweightfix"


def probe(root, gen):
    path = f"{root}/{gen}/Self_Play_Data.h5"
    try:
        f = h5.File(path, mode="r")
    except OSError as e:
        print(f"  gen {gen}: OSError {e}")
        return
    with f:
        keys = list(f.keys())
        stats = np.array(f["game_stats"])
        n_sets = (len(keys) - 1) // 3
        n_games = int(stats[2])
        aug = n_sets // n_games if n_games else -1
        b = np.array(f["boards_0"])
        p = np.array(f["policies_0"])
        v = np.array(f["values_0"])
        print(f"  gen {gen}: stats={stats.tolist()} n_sets={n_sets} n_games={n_games} aug={aug}")
        print(f"    boards_0 {b.shape} {b.dtype} range[{b.min()},{b.max()}] "
              f"policies_0 {p.shape} {p.dtype} values_0 {v.shape} {v.dtype}")
        print(f"    values_0 = {np.round(v.ravel(), 4).tolist()}")
        # board channel semantics on the first ply
        print(f"    boards_0[0] ch0 uniq={np.unique(b[0][:, :, 0])} ch3 nonzero={int(np.count_nonzero(b[0][:, :, 3]))}")
        print(f"    boards_0[1] ch0 uniq={np.unique(b[1][:, :, 0])} ch3 nonzero={int(np.count_nonzero(b[1][:, :, 3]))}")
        # last ply board
        print(f"    boards_0[-1] ch3 nonzero={int(np.count_nonzero(b[-1][:, :, 3]))} len={len(b)}")
        # is boards_1 the mirror of boards_0?
        if n_sets > 1:
            b1 = np.array(f["boards_1"])
            v1 = np.array(f["values_1"])
            same_v = np.allclose(v.ravel(), v1.ravel()) if v1.shape == v.shape else False
            mirror = np.array_equal(b1, b[:, :, ::-1, :]) if b1.shape == b.shape else False
            print(f"    boards_1 is column-mirror of boards_0: {mirror}; values equal: {same_v}")


def decomp(root, gen, n_games_probe=200):
    """values = w*z + (1-w)*q. z is +-1/0 constant per game (from the mover's view it
    alternates sign). Recover q for candidate w and check |q|<=1."""
    path = f"{root}/{gen}/Self_Play_Data.h5"
    try:
        f = h5.File(path, mode="r")
    except OSError as e:
        print(f"  decomp gen {gen}: OSError {e}")
        return
    with f:
        stats = np.array(f["game_stats"])
        n_sets = (len(f.keys()) - 1) // 3
        n_games = int(stats[2])
        aug = max(1, n_sets // max(1, n_games))
        for w in (1.0, 0.8, 0.6, 0.5):
            bad = 0
            tot = 0
            zmag = []
            for g in range(min(n_games, n_games_probe)):
                v = np.array(f[f"values_{g*aug}"]).ravel().astype(np.float64)
                T = len(v)
                # z from the perspective of the player to move at ply t alternates:
                # z_t = (+-1) * (-1)**t.  We don't know the outcome, so try all three.
                best = None
                for z0 in (-1.0, 0.0, 1.0):
                    z = z0 * ((-1.0) ** np.arange(T))
                    if w == 0:
                        continue
                    q = (v - w * z) / (1.0 - w) if w < 1.0 else np.zeros(T)
                    viol = 0 if w == 1.0 else int(np.sum(np.abs(q) > 1.0 + 1e-6))
                    if w == 1.0:
                        viol = int(np.sum(np.abs(v - z) > 1e-4))
                    if best is None or viol < best:
                        best = viol
                bad += best
                tot += T
                zmag.append(abs(v[0]))
            print(f"  gen {gen} w={w}: out-of-range/mismatch plies {bad}/{tot} = {bad/max(1,tot):.4f}")


if __name__ == "__main__":
    for root, name, gens in ((RUN2, "run2", [0, 8, 12, 19]), (RUN3, "run3", [0, 8, 12, 15])):
        print(f"=== {name} ===")
        for g in gens:
            probe(root, g)
        print(f"--- {name} value decomposition ---")
        for g in gens:
            decomp(root, g)
