"""Is the server's adaptive batching wait an artificial barrier? Measure it.

Real Server + real Parallelized_Session over real shared memory, 10 worker
processes that mimic search cadence: ~1.2ms of numpy busy-work between evals
(matching the measured 1 server trip per ~1.3ms of search).

Policies:
  adaptive : production controller (compute_wait_time, 5us..6ms, starts at 10ms)
  greedy   : dispatch immediately with whatever has arrived (wait_time = 0)
  fixed1ms : constant 1ms collection window

Reports per policy: aggregate evals/s, per-trip latency mean/p50/p95, and the
server's own batch-size debug stats (GZ_SERVER_DEBUG).
"""
import os
import sys
import time
import multiprocessing as mp
from pathlib import Path

ROOT = Path(r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np

MODEL = ROOT / "Connect4" / "Grok_Zero_Train" / "1" / "model.onnx"
N_WORKERS = 10
DURATION = 12.0
THINK_S = 0.0012


def run_server(policy, batched_in, batched_out, shms, request_ready, response_events):
    os.environ["GZ_SERVER_DEBUG"] = "1"
    from Client_Server import Server

    class TunedServer(Server):
        def __init__(self, *a, forced_wait=None, **kw):
            super().__init__(*a, **kw)
            self.forced_wait = forced_wait
            if forced_wait is not None:
                self.wait_time = forced_wait

        def compute_wait_time(self, alpha, avg_request_rate, min_wait=5e-6):
            if self.forced_wait is not None:
                return self.forced_wait
            return super().compute_wait_time(alpha, avg_request_rate, min_wait)

    forced = {"adaptive": None, "greedy": 0.0, "fixed1ms": 1e-3}[policy]
    srv = TunedServer(batched_in, batched_out, shms,
                      ["DmlExecutionProvider", "CPUExecutionProvider"],
                      MODEL.as_posix(), 1e-3, request_ready, response_events,
                      forced_wait=forced)
    srv.start()


def run_worker(wid, shm, single_in, single_out, request_ready, response_event,
               duration, out_q, start_evt):
    from Client_Server import Parallelized_Session
    sess = Parallelized_Session(wid, shm, single_in, single_out,
                                request_ready, response_event)
    x = np.random.uniform(-1, 2, (1, 6, 7, 4)).astype(np.float32)
    m = np.random.rand(48, 48).astype(np.float32)
    for _ in range(50):
        sess.run(["policy", "value"], {"inputs": x})

    start_evt.wait()
    lat, n = [], 0
    end = time.perf_counter() + duration
    while time.perf_counter() < end:
        t_think = time.perf_counter() + THINK_S      # search stand-in
        while time.perf_counter() < t_think:
            m = m @ m * 1e-3
        t0 = time.perf_counter()
        sess.run(["policy", "value"], {"inputs": x})
        lat.append(time.perf_counter() - t0)
        n += 1
    lat = np.array(lat) * 1000
    out_q.put((wid, n, float(lat.mean()), float(np.percentile(lat, 50)),
               float(np.percentile(lat, 95))))


def main():
    from Client_Server import create_shared_memory, create_events, convert_to_single_info

    batched_in = {"inputs": [[-1, 6, 7, 4], np.float32]}
    batched_out = {"policy": [-1, 7], "value": [-1, 1]}
    single_in = convert_to_single_info(batched_in)
    single_out = convert_to_single_info(batched_out)

    print(f"{N_WORKERS} workers, {THINK_S*1e3:.1f}ms think time between evals, "
          f"{DURATION:.0f}s per policy, DirectML\n")

    for policy in ("adaptive", "greedy", "fixed1ms"):
        shms = create_shared_memory(batched_in, batched_out, N_WORKERS)
        request_ready, response_events = create_events(N_WORKERS)
        srv = mp.Process(target=run_server,
                         args=(policy, batched_in, batched_out, shms,
                               request_ready, response_events))
        srv.start()

        out_q = mp.Queue()
        start_evt = mp.Event()
        ws = [mp.Process(target=run_worker,
                         args=(i, shms[i], single_in, single_out, request_ready,
                               response_events[i], DURATION, out_q, start_evt))
              for i in range(N_WORKERS)]
        for w in ws:
            w.start()
        time.sleep(8.0)          # let all sessions warm up
        start_evt.set()

        rows = [out_q.get(timeout=DURATION + 60) for _ in range(N_WORKERS)]
        for w in ws:
            w.join()
        srv.terminate()
        srv.join()
        for shm in shms:
            shm.close()
            shm.unlink()

        total = sum(r[1] for r in rows)
        print(f"== {policy:<9} aggregate {total / DURATION:8.0f} evals/s | "
              f"per-trip mean {np.mean([r[2] for r in rows]):6.3f} ms  "
              f"p50 {np.mean([r[3] for r in rows]):6.3f}  "
              f"p95 {np.mean([r[4] for r in rows]):6.3f}", flush=True)
        time.sleep(1)


if __name__ == "__main__":
    main()
