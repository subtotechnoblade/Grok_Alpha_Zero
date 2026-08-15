"""Full profile of ONE production worker against the REAL inference server.

Layout mirrors run_self_play exactly, except the worker runs in THIS process so
cProfile sees it: server = child process (DirectML), worker = inline self_play_task.

Phases:
  1. round-trip microbenchmark: 1000 timed session.run() calls through the real
     shm+event path (this is the per-eval latency, isolated)
  2. cProfile of self_play_task playing N full games (production config: memo,
     h5 writes, dirichlet, tau schedule, use_njit resolved as production does)

usage: profile_worker.py [n_games] [use_njit: prod|true]
"""
import cProfile
import io
import os
import pstats
import shutil
import sys
import time
import multiprocessing as mp
from pathlib import Path

ROOT = Path(r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np

SRC_MODEL = ROOT / "Connect4" / "Grok_Zero_Train" / "1" / "model.onnx"
WORK = ROOT / "tmp" / "prof_run" / "1"


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    njit_mode = sys.argv[2] if len(sys.argv) > 2 else "prod"

    from Connect4.game import Connect4
    from Connect4.config import train_config, build_config, optimizer_config
    from pipeline import Make_Dataset_File
    from Self_Play import self_play_task, convert_shape
    from Client_Server import (Parallelized_Session, create_shared_memory,
                               create_events, start_server, convert_to_single_info)

    cfg = train_config.replace(num_workers=1, use_tensorrt=False)
    if njit_mode == "true":
        cfg = cfg.replace(use_njit=True)
    cache_depth = os.environ.get("CACHE_DEPTH")
    if cache_depth:
        cfg = cfg.replace(max_cache_depth=int(cache_depth))
    # else: leave None -> resolves False on Windows, exactly like production

    shutil.rmtree(WORK.parent, ignore_errors=True)
    WORK.mkdir(parents=True)
    shutil.copy2(SRC_MODEL, WORK / "model.onnx")
    Make_Dataset_File(WORK.as_posix())

    game = Connect4()
    inputs_shape = game.get_input_state().shape
    policy_shape = game.policy_shape
    del game

    batched_in = {"inputs": [[-1, *inputs_shape], np.float32]}
    batched_out = {"policy": [-1, *policy_shape], "value": [-1, 1]}
    shms = create_shared_memory(batched_in, batched_out, 1)
    request_ready, response_events = create_events(1)

    providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
    server = mp.Process(target=start_server,
                        args=(batched_in, batched_out, shms, providers,
                              (WORK / "model.onnx").as_posix(), 1e-3,
                              request_ready, response_events))
    server.start()

    single_in = convert_to_single_info(batched_in)
    single_out = convert_to_single_info(batched_out)

    # ---- phase 1: raw round-trip latency through the real path ----
    sess = Parallelized_Session(0, shms[0], single_in, single_out,
                                request_ready, response_events[0])
    x = np.random.uniform(-1, 2, (1, *inputs_shape)).astype(np.float32)
    for _ in range(200):
        sess.run(["policy", "value"], {"inputs": x})
    lat = []
    for _ in range(1000):
        t0 = time.perf_counter()
        sess.run(["policy", "value"], {"inputs": x})
        lat.append(time.perf_counter() - t0)
    lat = np.array(lat) * 1000
    print(f"ROUND TRIP (1 worker, DirectML): mean {lat.mean():.3f} ms   "
          f"p50 {np.percentile(lat, 50):.3f}   p95 {np.percentile(lat, 95):.3f}   "
          f"max {lat.max():.2f}", flush=True)
    print(f"  spin-path wakeups {sess.wait_wakeups}, timeouts {sess.wait_timeouts}\n")

    # ---- phase 2: cProfile the real worker task ----
    lock = mp.Lock()
    claimed = mp.Value("i", 0)
    progress = mp.Value("i", 0)
    stats_q = mp.Queue()

    t0 = time.perf_counter()
    prof = cProfile.Profile()
    prof.enable()
    self_play_task(0, shms[0], Connect4, single_in, single_out,
                   build_config, cfg, lock, WORK.as_posix(), 1,
                   claimed, n_games, progress, stats_q,
                   request_ready, response_events[0])
    prof.disable()
    wall = time.perf_counter() - t0

    wid, memo = stats_q.get(timeout=5)
    print(f"njit={njit_mode}: {n_games} games in {wall:.1f}s = {wall / n_games:.2f} s/game "
          f"(cProfile overhead included)")
    if memo:
        print(f"memo: {memo['hits']}h/{memo['misses']}m ({memo['hit_rate']:.0%})\n")

    for sort in ("cumulative", "tottime"):
        s = io.StringIO()
        ps = pstats.Stats(prof, stream=s).sort_stats(sort)
        ps.print_stats(18)
        lines = s.getvalue().splitlines()
        keep = [l for l in lines if l.strip()][:26]
        print(f"---- top by {sort} ----")
        print("\n".join(keep))
        print()

    server.terminate()
    for shm in shms:
        shm.close()
        shm.unlink()
    shutil.rmtree(WORK.parent, ignore_errors=True)


if __name__ == "__main__":
    main()
