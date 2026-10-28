"""Where the time goes in batched self play: server loop phases and worker round trips.
python -m tests.profile_server <model_dir> [workers] [seconds] [leaves_per_batch]"""
import multiprocessing as mp
import sys
import time

import numpy as np

from tests.mcts_scenarios import ROOT
from tests.bench_search_speed import providers_for, model_path, search_loop, gpu_util_sampler, STATS
from Gomoku.game import Gomoku

SERVER = ("loops", "batches", "rows", "gather", "run", "scatter", "idle_scan", "wait")
WORKER = ("requests", "request_time", "elapsed")


def timed_server(spec, slots, providers, file_path, request_ready, response_events, out):
    import signal
    from Client_Server import Batch_Server, make_engine
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    engine = make_engine(file_path, providers, spec.max_rows * len(slots))
    server = Batch_Server(spec, slots, engine, request_ready, response_events)
    acc = dict.fromkeys(SERVER, 0.0)
    clock = time.perf_counter
    while True:
        t0 = clock()
        routes, total = [], 0
        for end in server.ends:
            n = end.pending()
            if n:
                for name, rows in end.inputs(n).items():
                    server.batch[name][total:total + n] = rows
                routes.append((end, total, n))
                total += n
        t1 = clock()
        acc["loops"] += 1
        if not routes:
            acc["idle_scan"] += t1 - t0
            request_ready.wait(0.05)
            acc["wait"] += clock() - t1
        else:
            outputs = engine.run(server.output_names, {name: arr[:total] for name, arr in server.batch.items()})
            t2 = clock()
            for end, start, n in routes:
                o = end.outputs(n)
                for name, values in zip(server.output_names, outputs):
                    o[name][:] = values[start:start + n]
                end.respond()
            t3 = clock()
            acc["gather"] += t1 - t0
            acc["run"] += t2 - t1
            acc["scatter"] += t3 - t2
            acc["batches"] += 1
            acc["rows"] += total
        if acc["loops"] % 64 == 0:
            out[:] = [acc[key] for key in SERVER]


class TimedSession:
    def __init__(self, inner):
        self.inner = inner
        self.input_dtype = inner.input_dtype
        self.requests = 0
        self.request_time = 0.0

    def run(self, output_names, input_feed, run_options=None):
        t0 = time.perf_counter()
        result = self.inner.run(output_names, input_feed)
        self.request_time += time.perf_counter() - t0
        self.requests += 1
        return result


def worker(spec, shm, request_ready, response_ready, seconds, k, counter, stats, barrier, wstats):
    from Client_Server import Batched_Session
    inner = Batched_Session(spec, shm, request_ready, response_ready)
    sess = TimedSession(inner)
    original_wait = barrier.wait

    def wait_and_reset():
        original_wait()
        sess.requests, sess.request_time = 0, 0.0
        wait_and_reset.t0 = time.perf_counter()

    class B:
        wait = staticmethod(wait_and_reset)

    search_loop(sess, seconds, k, counter, stats, B)
    elapsed = time.perf_counter() - wait_and_reset.t0
    with wstats.get_lock():
        wstats[0] += sess.requests
        wstats[1] += sess.request_time
        wstats[2] += elapsed
    inner.client.close()
    shm.close()


if __name__ == "__main__":
    from Client_Server import create_wakeups
    from Transport import ChannelSpec, create_slots
    model_dir = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    game = Gomoku()
    state = game.get_input_state()
    pmax = workers * k
    wire = state.dtype if state.dtype.kind in "biu" else np.float32
    spec = ChannelSpec({"inputs": (state.shape, wire)},
                       {"policy": (game.policy_shape, np.float32), "value": ((1,), np.float32)}, max_rows=k)
    shms = create_slots(spec, workers)
    request_ready, response_events = create_wakeups(workers)
    server_out = mp.Array("d", len(SERVER))
    counter, stats = mp.Value("q", 0), mp.Array("q", len(STATS))
    wstats = mp.Array("d", len(WORKER))
    barrier = mp.Barrier(workers + 1)
    server = mp.Process(target=timed_server, args=(spec, shms, providers_for(model_dir, state.shape, pmax),
                                                    model_path(model_dir, pmax), request_ready, response_events,
                                                    server_out))
    server.start()
    procs = [mp.Process(target=worker, args=(spec, shms[i], request_ready, response_events[i], seconds, k,
                                              counter, stats, barrier, wstats)) for i in range(workers)]
    for p in procs:
        p.start()
    barrier.wait()
    before = list(server_out[:])
    t0 = time.perf_counter()
    util = gpu_util_sampler(seconds)
    for p in procs:
        p.join()
    elapsed = time.perf_counter() - t0
    after = list(server_out[:])
    server.terminate()
    server.join()
    for shm in shms:
        shm.close()
        shm.unlink()
    d = dict(zip(SERVER, (a - b for a, b in zip(after, before))))
    span = d["gather"] + d["run"] + d["scatter"] + d["idle_scan"] + d["wait"]
    batches = max(d["batches"], 1)
    print(f"RESULT workers={workers} K={k} sims/s={counter.value / elapsed:,.0f} gpu_util_mean={np.mean(util):.0f}%")
    print(f"server: {int(d['batches'])} batches, {d['rows'] / batches:.1f} rows/batch, "
          f"{d['batches'] / span:,.0f} batches/s over {span:.1f}s accounted")
    for key in ("gather", "run", "scatter", "idle_scan", "wait"):
        print(f"  {key:9s} {100 * d[key] / span:5.1f}%  {1e6 * d[key] / batches:7.0f} us/batch")
    n, rt, el = wstats[:]
    print(f"workers: {n / el * workers:,.0f} requests/s total, round trip {1e6 * rt / max(n, 1):.0f} us, "
          f"waiting {100 * rt / el:.0f}% of the time")
