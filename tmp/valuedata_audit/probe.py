"""Probe: confirm h5 layout + target decomposition per run/gen."""
import sys, numpy as np, h5py as h5

RUN2 = r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero\tmp\run2_archive_gen0-19_oldvaluehead"
RUN3 = r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero\tmp\run3_archive_gen0-15_bottleneck_preweightfix"


def probe(run, gen):
    path = f"{run}\\{gen}\\Self_Play_Data.h5"
    try:
        f = h5.File(path, mode="r")
    except OSError as e:
        print(f"  OPEN FAIL {path}: {e}")
        return
    with f:
        keys = list(f.keys())
        n_sets = (len(keys) - 1) // 3
        gs = np.array(f["game_stats"])
        n_games = int(gs[2])
        aug = n_sets // n_games if n_games else 0
        b0 = f["boards_0"]
        v0 = f["values_0"]
        p0 = f["policies_0"]
        print(f"  gen {gen}: n_sets={n_sets} n_games={n_games} aug={aug} "
              f"game_stats={gs.tolist()}")
        print(f"    boards_0 {b0.shape} {b0.dtype} | policies_0 {p0.shape} {p0.dtype} "
              f"| values_0 {v0.shape} {v0.dtype}")
        # ply lengths of the first few unmirrored games
        lens = [f[f"values_{g*aug}"].shape[0] for g in range(min(12, n_games))]
        print(f"    first ply-lengths (unmirrored): {lens}")
        print(f"    values_0 flat: {np.array(v0).ravel().round(3).tolist()}")
        # last-ply target across games -> should be near +1 side for decided games
        lastv = np.array([f[f"values_{g*aug}"][-1, 0] for g in range(min(400, n_games))])
        print(f"    last-ply target: mean={lastv.mean():.3f} min={lastv.min():.3f} "
              f"max={lastv.max():.3f}  frac>0.5: {(lastv > .5).mean():.3f}")
        allv = np.concatenate([np.array(f[f"values_{g*aug}"]).ravel()
                               for g in range(min(300, n_games))])
        print(f"    all targets: min={allv.min():.4f} max={allv.max():.4f} "
              f"n_exact_pm1={np.mean(np.abs(np.abs(allv) - 1) < 1e-6):.3f} "
              f"n_exact_0={np.mean(np.abs(allv) < 1e-9):.4f}")
        u = np.unique(np.round(allv, 6))
        print(f"    unique target values (first 12 of {len(u)}): {u[:12].tolist()}")


for run, name, gens in ((RUN2, "run2", [0, 3, 5, 8, 10, 15, 19]),
                        (RUN3, "run3", [0, 3, 5, 10, 15])):
    print(f"== {name} ==")
    for g in gens:
        probe(run, g)
