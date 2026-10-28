"""Self-play search speed on Gomoku with a real TensorRT model.
python -m tests.bench_search_speed <model_dir> local|server|batched [workers] [seconds] [leaves_per_batch] [sims]
local: one TensorRT session per worker. server: the old Parallelized_Session server (K=1 only).
batched: Batch_Server with int8 on the wire.
model_dir holds model.onnx; each TensorRT profile gets its own TRT_cache_<max>/ (use a scratch copy)."""
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np

from tests.mcts_scenarios import ROOT
from Gomoku.game import Gomoku
from Self_Play import convert_shape

SIMS = 600
STATS = ("batches", "network_rows", "collisions", "terminal")


def providers_for(model_dir, shape, profile_max):
    s = convert_shape(shape)
    return [('TensorrtExecutionProvider', {
        "trt_engine_cache_enable": True,
        "trt_dump_ep_context_model": True,
        "trt_builder_optimization_level": 5,
        "trt_auxiliary_streams": 0,
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": f"{model_dir}/",
        "trt_ep_context_file_path": f"{model_dir}/TRT_cache_{profile_max}/",
        "trt_profile_min_shapes": f"inputs:1x{s}",
        "trt_profile_max_shapes": f"inputs:{profile_max}x{s}",
        "trt_profile_opt_shapes": f"inputs:{profile_max}x{s}",
        "trt_fp16_enable": True}), 'CUDAExecutionProvider', 'CPUExecutionProvider']


def model_path(model_dir, profile_max):
    ctx = f"{model_dir}/TRT_cache_{profile_max}/model_ctx.onnx"
    return ctx if os.path.exists(ctx) else f"{model_dir}/model.onnx"


def search_loop(session, seconds, k, counter, stats, barrier):
    from MCTS import MCTS
    from Session_Cache import Cache_Wrapper
    np.random.seed()
    session = Cache_Wrapper(session, 3)
    totals = dict.fromkeys(STATS, 0)

    def new_mcts():
        game = Gomoku()
        return game, MCTS(game, session, c_puct_init=2.0, fpu_reduction=0.2, use_dirichlet=True,
                          dirichlet_alpha=0.05, tau=1.0, leaves_per_batch=k)

    def collect(mcts):
        for key in STATS:
            totals[key] += mcts.search_stats[key]

    game, mcts = new_mcts()
    mcts.run(iteration_limit=SIMS, use_bar=False)
    game, mcts = new_mcts()
    barrier.wait()
    end = time.perf_counter() + seconds
    sims = 0
    while time.perf_counter() < end:
        if game.action_history and game.check_win() != -2:
            collect(mcts)
            game, mcts = new_mcts()
        move, _ = mcts.run(iteration_limit=SIMS, use_bar=False)
        sims += SIMS
        game.do_action(move)
        mcts.prune_tree(move)
    collect(mcts)
    with counter.get_lock():
        counter.value += sims
    with stats.get_lock():
        for i, key in enumerate(STATS):
            stats[i] += totals[key]


def local_worker(model_dir, seconds, k, counter, stats, barrier):
    import onnxruntime as rt
    from Client_Server import preload_gpu_dlls
    preload_gpu_dlls()
    shape = Gomoku().get_input_state().shape
    sess = rt.InferenceSession(model_path(model_dir, k), providers=providers_for(model_dir, shape, k))
    search_loop(sess, seconds, k, counter, stats, barrier)


def server_worker(worker_id, shm, in_info, out_info, request_ready, response_ready, seconds, counter, stats,
                  barrier):
    from Client_Server import Parallelized_Session
    sess = Parallelized_Session(worker_id, shm, in_info, out_info, request_ready, response_ready)
    search_loop(sess, seconds, 1, counter, stats, barrier)


def batched_worker(spec, shm, request_ready, response_ready, seconds, k, counter, stats, barrier):
    from Client_Server import Batched_Session
    sess = Batched_Session(spec, shm, request_ready, response_ready)
    search_loop(sess, seconds, k, counter, stats, barrier)
    sess.client.close()
    shm.close()


def gpu_util_sampler(seconds):
    samples = []
    end = time.time() + seconds
    while time.time() < end:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.strip()
        if out:
            samples.append(int(out.splitlines()[0]))
        time.sleep(1.0)
    return samples


if __name__ == "__main__":
    model_dir, mode = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    seconds = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0
    k = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    SIMS = int(sys.argv[6]) if len(sys.argv) > 6 else SIMS
    ctx = mp
    counter = ctx.Value("q", 0)
    stats = ctx.Array("q", len(STATS))
    barrier = ctx.Barrier(workers + 1)
    game = Gomoku()
    state = game.get_input_state()
    shape = state.shape
    procs, server = [], None
    shms = []
    if mode == "local":
        procs = [ctx.Process(target=local_worker, args=(model_dir, seconds, k, counter, stats, barrier))
                 for _ in range(workers)]
    elif mode == "server":
        from Client_Server import start_server, create_shared_memory, create_events, convert_to_single_info
        pmax = workers
        in_info = {"inputs": [[-1, *shape], np.float32]}
        out_info = {"policy": [-1, 225], "value": [-1, 1]}
        single_in, single_out = convert_to_single_info(in_info), convert_to_single_info(out_info)
        shms = create_shared_memory(in_info, out_info, workers)
        request_ready, response_events = create_events(workers)
        server = ctx.Process(target=start_server,
                             args=({"inputs": [[-1, *shape], np.float32]}, {"policy": [-1, 225], "value": [-1, 1]},
                                   shms, providers_for(model_dir, shape, pmax), model_path(model_dir, pmax),
                                   0.001, request_ready, response_events))
        server.start()
        procs = [ctx.Process(target=server_worker, args=(i, shms[i], single_in, single_out, request_ready,
                                                          response_events[i], seconds, counter, stats, barrier))
                 for i in range(workers)]
    else:
        from Client_Server import start_batch_server, create_wakeups
        from Transport import ChannelSpec, create_slots
        pmax = workers * k
        wire = state.dtype if state.dtype.kind in "biu" else np.float32
        spec = ChannelSpec({"inputs": (shape, wire)},
                           {"policy": (game.policy_shape, np.float32), "value": ((1,), np.float32)}, max_rows=k)
        shms = create_slots(spec, workers)
        request_ready, response_events = create_wakeups(workers)
        server = ctx.Process(target=start_batch_server,
                             args=(spec, shms, providers_for(model_dir, shape, pmax), model_path(model_dir, pmax),
                                   request_ready, response_events))
        server.start()
        procs = [ctx.Process(target=batched_worker, args=(spec, shms[i], request_ready, response_events[i],
                                                           seconds, k, counter, stats, barrier))
                 for i in range(workers)]
    del game, state
    for p in procs:
        p.start()
    barrier.wait()
    t0 = time.perf_counter()
    util = gpu_util_sampler(seconds)
    for p in procs:
        p.join()
    elapsed = time.perf_counter() - t0
    if server is not None:
        server.terminate()
        server.join()
        for shm in shms:
            shm.close()
            shm.unlink()
    batches, rows, collisions, terminal = stats[:]
    print(f"RESULT mode={mode} workers={workers} K={k} sims={SIMS} sims/s={counter.value / elapsed:,.0f} "
          f"gpu_util_mean={np.mean(util) if util else float('nan'):.0f}% "
          f"rows/batch={rows / max(batches, 1):.2f} collisions/sim={collisions / max(counter.value, 1):.3f} "
          f"terminal/sim={terminal / max(counter.value, 1):.3f} over {elapsed:.0f}s", flush=True)
