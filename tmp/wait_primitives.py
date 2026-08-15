"""Option 2 (spin-then-sleep) vs option 3 (mp.Event) - measure the primitives.

The choice hinges on two numbers this box actually delivers:
  A. what does time.sleep(x) really sleep for on Windows? (granularity decides whether
     a backoff loop adds meaningful latency)
  B. what does an mp.Event round trip cost between processes? (decides whether proper
     blocking is affordable per inference call)

Reference point from tmp/spin_fraction.py: workers currently wait ~8,600 us per call
under load, and a standalone batch-6 DirectML call is ~630 us.
"""
import multiprocessing as mp
import os
import sys
import time

import numpy as np

REQUESTS = [0, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]


def sleep_granularity():
    print("A. time.sleep() actual duration on this platform")
    print(f"   {'requested':>12} {'median actual':>15} {'min':>10} {'overshoot':>11}")
    for req in REQUESTS:
        ts = []
        for _ in range(300):
            t0 = time.perf_counter()
            time.sleep(req)
            ts.append(time.perf_counter() - t0)
        med, mn = np.median(ts) * 1e6, np.min(ts) * 1e6
        over = "-" if req == 0 else f"{med / (req * 1e6):.1f}x"
        print(f"   {req * 1e6:>10.0f}us {med:>13.1f}us {mn:>8.1f}us {over:>11}")


def _pong(req_ev, resp_ev, n):
    for _ in range(n):
        req_ev.wait()
        req_ev.clear()
        resp_ev.set()


def event_roundtrip():
    print("\nB. mp.Event round trip between two processes")
    N = 2000
    req_ev, resp_ev = mp.Event(), mp.Event()
    p = mp.Process(target=_pong, args=(req_ev, resp_ev, N))
    p.start()
    time.sleep(1.0)
    ts = []
    for _ in range(N):
        t0 = time.perf_counter()
        req_ev.set()
        resp_ev.wait()
        resp_ev.clear()
        ts.append(time.perf_counter() - t0)
    p.join(10)
    if p.is_alive():
        p.terminate()
    a = np.array(ts) * 1e6
    print(f"   median {np.median(a):.1f}us   p90 {np.percentile(a, 90):.1f}us   "
          f"min {a.min():.1f}us   (set + wake + set + wake)")
    return float(np.median(a))


def _spin_pong(shm_name, n):
    from multiprocessing.shared_memory import SharedMemory
    shm = SharedMemory(name=shm_name)
    arr = np.ndarray(shape=(8,), dtype=np.float32, buffer=shm.buf)
    for _ in range(n):
        while arr[0] != 1.0:
            pass
        arr[0] = 0.0


def spin_roundtrip():
    print("\nC. pure spin round trip over shared memory (what the code does now)")
    from multiprocessing.shared_memory import SharedMemory
    N = 2000
    shm = SharedMemory(create=True, size=32)
    arr = np.ndarray(shape=(8,), dtype=np.float32, buffer=shm.buf)
    arr[0] = 0.0
    p = mp.Process(target=_spin_pong, args=(shm.name, N))
    p.start()
    time.sleep(1.0)
    ts = []
    for _ in range(N):
        t0 = time.perf_counter()
        arr[0] = 1.0
        while arr[0] != 0.0:
            pass
        ts.append(time.perf_counter() - t0)
    p.join(10)
    if p.is_alive():
        p.terminate()
    shm.close()
    shm.unlink()
    a = np.array(ts) * 1e6
    print(f"   median {np.median(a):.1f}us   p90 {np.percentile(a, 90):.1f}us "
          f"  (uncontended, both processes have a free core)")
    return float(np.median(a))


def main():
    sleep_granularity()
    ev = event_roundtrip()
    sp = spin_roundtrip()

    print("\n" + "=" * 68)
    print("what each option costs PER INFERENCE CALL")
    print(f"   current spin (uncontended)          : {sp:8.1f} us")
    print(f"   option 3, mp.Event                  : {ev:8.1f} us")
    print("   option 2, spin-then-sleep           : depends on the granularity above")
    print()
    print("measured context: workers currently wait ~8,600 us/call under load,")
    print("and a standalone batch-6 DirectML call takes ~630 us.")


if __name__ == "__main__":
    main()
