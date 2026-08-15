"""Is the cpu_count()//2 worker cap still needed?

The cap dates from when a worker was spawned PER GAME: hundreds of short-lived
processes, each re-importing everything and each racing to compile/write numba's
on-disk cache with no cross-process lock. Workers are now persistent - one process
per worker for the whole generation, chewing through its own subset of games - so
both the import storm and the cache race should be gone.

This runs the real run_self_play at several worker counts, bypassing the cap, and
reports wall time, throughput and whether the run stayed correct.

Run:
  C:\\Users\\TryHarder\\anaconda3\\envs\\tensorflow_updated\\python.exe tmp/worker_scaling.py [games] [sims]
"""
import os
import shutil
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    os.chdir(ROOT)
    import multiprocessing as mp
    import numpy as np
    import h5py as h5
    from Connect4.Connect4 import Connect4, build_config, train_config, optimizer_config
    import Self_Play as SP

    GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    SIMS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    USE_GPU = "--gpu" in sys.argv
    COUNTS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 and \
        sys.argv[3][0].isdigit() else [2, 4, 6, 8, 10, 12]
    real_cpu = mp.cpu_count()

    base = ROOT.replace("\\", "/") + "/tmp/worker_scaling"
    shutil.rmtree(base, ignore_errors=True)

    src = f"{ROOT}/tmp/dml_bench/model.onnx"
    if not os.path.exists(src):
        raise SystemExit("run tmp/windows_dml_bench.py first to export a model")

    print(f"logical cores {real_cpu} | cap would be {real_cpu // 2} | "
          f"{GAMES} games x {SIMS} sims per config")
    print(f"start method: {mp.get_start_method()} | inference: "
          f"{'DirectML (iGPU)' if USE_GPU else 'CPU EP'}")
    print(f"{'workers':>8} {'wall s':>9} {'games/s':>9} {'s/game/worker':>14} "
          f"{'games':>7} {'status':>10}")

    baseline = None
    for n in COUNTS:
        folder = f"{base}/{n}"
        os.makedirs(folder, exist_ok=True)
        shutil.copy(src, f"{folder}/model.onnx")
        with h5.File(f"{folder}/Self_Play_Data.h5", "w", libver="latest") as f:
            f.create_dataset("game_stats", data=np.zeros(6, dtype=np.int64),
                             maxshape=(6,))

        tc = dict(train_config)
        tc["max_actions"] = 42
        tc["games_per_generation"] = GAMES
        tc["num_workers"] = n
        tc["MCTS_iteration_limit"] = SIMS
        tc["use_inference_server"] = True
        tc["use_gpu"] = USE_GPU
        tc["use_tensorrt"] = False

        # bypass the cap so we can see what it was protecting against
        mp.cpu_count = lambda: 1024
        SP.mp.cpu_count = lambda: 1024
        t0 = time.perf_counter()
        status = "ok"
        try:
            SP.run_self_play(Connect4, (build_config, tc, optimizer_config), folder, 1)
        except Exception as e:
            status = f"{type(e).__name__}"
        dt = time.perf_counter() - t0
        mp.cpu_count = SP.mp.cpu_count = (lambda c=real_cpu: c)

        try:
            with h5.File(f"{folder}/Self_Play_Data.h5", "r") as f:
                played = int(f["game_stats"][2])
                triples = (len(f.keys()) - 1) // 3
            if played != GAMES or triples != GAMES * 2:
                status = f"BAD {played}g/{triples}t"
        except Exception:
            played, status = 0, status if status != "ok" else "no h5"

        gps = played / dt if dt else 0
        baseline = baseline or gps
        print(f"{n:>8} {dt:>9.1f} {gps:>9.2f} {dt * n / max(1, played):>14.2f} "
              f"{played:>7} {status:>10}")

    print()
    print("s/game/worker rising = workers are contending; flat = still scaling")
    shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
