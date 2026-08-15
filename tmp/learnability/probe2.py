"""Determine q_weight per generation empirically.

Self_Play.py:162  target_values = (1 - q_weight) * z + q_weight * q
In a DRAWN game z == 0 for every ply, so v == q_weight * q and |v| <= q_weight.
Pooling many drawn games, max|v| saturates at q_weight (q does reach +-1 mid-game).
"""
import numpy as np
import h5py as h5

RUN2 = "C:/Users/TryHarder/Desktop/Pythoncode/Grok_Zero/tmp/run2_archive_gen0-19_oldvaluehead"
RUN3 = "C:/Users/TryHarder/Desktop/Pythoncode/Grok_Zero/tmp/run3_archive_gen0-15_bottleneck_preweightfix"


def gen_info(root, gen):
    path = f"{root}/{gen}/Self_Play_Data.h5"
    try:
        with h5.File(path, mode="r") as f:
            keys = set(f.keys())
            stats = np.array(f["game_stats"])
            n_sets = (len(keys) - 1) // 3
            n_games = int(stats[2])
            have = sum(1 for i in range(n_sets) if f"boards_{i}" in keys)
            aug = n_sets // n_games if n_games and n_sets % n_games == 0 else None
            drawmax, drawn, plies, allmax = 0.0, 0, 0, 0.0
            v0s = []
            for i in range(n_sets):
                k = f"values_{i}"
                if k not in keys:
                    continue
                v = np.array(f[k]).ravel().astype(np.float64)
                if len(v) == 0:
                    continue
                plies += len(v)
                allmax = max(allmax, float(np.abs(v).max()))
                v0s.append(float(v[0]))
                if abs(v[-1]) < 0.5:          # drawn game: last ply value ~ 0
                    drawn += 1
                    drawmax = max(drawmax, float(np.abs(v).max()))
            return dict(stats=stats.tolist(), n_sets=n_sets, n_games=n_games, aug=aug,
                        have=have, drawn=drawn, drawmax=drawmax, allmax=allmax,
                        plies=plies, v0_absmax=float(np.abs(v0s).max()) if v0s else 0.0)
    except OSError as e:
        return dict(error=str(e)[:80])


if __name__ == "__main__":
    for root, name, gens in ((RUN2, "run2", range(20)), (RUN3, "run3", range(16))):
        print(f"=== {name} ===")
        for g in gens:
            d = gen_info(root, g)
            if "error" in d:
                print(f" gen {g:2d}: ERROR {d['error']}")
                continue
            qw = d["drawmax"]
            print(f" gen {g:2d}: games={d['n_games']:5d} sets={d['n_sets']:5d} present={d['have']:5d} "
                  f"aug={d['aug']} plies={d['plies']:7d} drawn_sets={d['drawn']:5d} "
                  f"max|v| in draws={qw:.4f}  max|v| all={d['allmax']:.4f}  max|v0|={d['v0_absmax']:.4f}")
