"""How much of a worker's time is spent spinning on the shared memory flag?

Parallelized_Session.run busy-waits in two loops with no sleep and no blocking call.
This replicates the real self-play setup - same shared memory, same server, real MCTS
- but swaps in a Parallelized_Session subclass that times both loops separately:

  send wait : waiting for our slot to be free (server still holds the previous batch)
  recv wait : waiting for the answer (inference latency)

Everything else the worker does (select/expand/backprop) is real search. So
  spin fraction = (send + recv) / total worker wall time
is the share of CPU a worker burns doing nothing but polling.

Run:
  ...\\tensorflow_updated\\python.exe tmp/spin_fraction.py [workers] [games] [sims] [--gpu]
"""
import os
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import multiprocessing as mp


class TimedSession:
    """Same protocol as Parallelized_Session.run, with the two spin loops timed.

    Deliberately a copy rather than a subclass calling super(): the timers have to sit
    INSIDE the loops, and the point is to measure the loops exactly as written.
    """

    def __init__(self, base):
        self.b = base
        self.send_ns = 0
        self.recv_ns = 0
        self.calls = 0

    def run(self, output_names, input_feed):
        import numpy as np
        b = self.b
        self.calls += 1

        t0 = time.perf_counter_ns()
        b._await_free_slot()
        data = np.concatenate([a.flatten() for a in input_feed.values()],
                              dtype=np.float32)
        b.shared_arr[1:1 + len(data)] = data
        b.shared_arr[0] = 1.0
        if b.request_ready is not None:
            b.request_ready.set()
        self.send_ns += time.perf_counter_ns() - t0

        t0 = time.perf_counter_ns()
        b._await_free_slot()
        self.recv_ns += time.perf_counter_ns() - t0
        raw = b.shared_arr[:]
        outputs = [None] * len(b.outputs_feed_info)
        start = 1
        for i, (shape, length) in enumerate(b.outputs_feed_info.values()):
            outputs[i] = np.array(raw[start:start + length], copy=True).reshape(shape)
            start += length
        return outputs


def worker(wid, shm_name, in_info, out_info, n_games, sims, q,
           request_ready=None, response_ready=None, spin_limit=500):
    os.chdir(ROOT)
    import numpy as np
    from multiprocessing.shared_memory import SharedMemory
    from Client_Server import Parallelized_Session
    from Connect4.Connect4 import Connect4, train_config
    from MCTS import MCTS

    np.random.seed(1000 + wid)
    shm = SharedMemory(name=shm_name)
    base = Parallelized_Session(wid, shm, in_info, out_info,
                                request_ready, response_ready, spin_limit)
    sess = TimedSession(base)

    t_start = time.perf_counter()
    cpu_start = time.process_time()   # CPU actually burned, the number that matters
    moves = 0
    for _ in range(n_games):
        g = Connect4()
        tree = MCTS(g, sess, use_njit=train_config["use_njit"],
                    c_puct_init=train_config["c_puct_init"], use_dirichlet=True,
                    dirichlet_alpha=train_config["dirichlet_alpha"], tau=1.0,
                    fast_find_win=False)
        winner = -2
        n = 0
        while winner == -2 and n < 42:
            a, _ = tree.run(int(sims * 1.5), use_bar=False)
            g.do_action(a)
            winner = g.check_win()
            if winner == -2:
                tree.prune_tree(a)
            n += 1
        moves += n
    total = time.perf_counter() - t_start
    cpu = time.process_time() - cpu_start
    q.put((wid, total, sess.send_ns / 1e9, sess.recv_ns / 1e9, sess.calls, moves,
           base.wait_timeouts, base.wait_wakeups, cpu))


def main():
    import numpy as np
    from Client_Server import (create_shared_memory, start_server,
                               convert_to_single_info, create_events)
    from Connect4.Connect4 import Connect4, build_config, train_config

    W = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    SIMS = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    USE_GPU = "--gpu" in sys.argv
    NO_EVENTS = "--spin" in sys.argv       # force the old pure-spin path for A/B
    SPIN = 500
    for a in sys.argv:
        if a.startswith("--spin-limit="):
            SPIN = int(a.split("=")[1])

    model = f"{ROOT}/tmp/dml_bench/model.onnx"
    game = Connect4()
    in_info = {"inputs": [[-1, *game.get_input_state().shape], np.float32]}
    out_info = {"policy": [-1, *game.policy_shape], "value": [-1, 1]}
    providers = (["DmlExecutionProvider", "CPUExecutionProvider"] if USE_GPU
                 else ["CPUExecutionProvider"])

    shms = create_shared_memory(dict(in_info), dict(out_info), W)
    single_in = convert_to_single_info(dict(in_info))
    single_out = convert_to_single_info(dict(out_info))

    req_ev, resp_evs = (None, [None] * W) if NO_EVENTS else create_events(W)
    server = mp.Process(target=start_server,
                        args=(dict(in_info), dict(out_info), shms, providers,
                              model, 0.001, req_ev, resp_evs))
    server.start()
    time.sleep(6)

    print(f"{W} workers | {GAMES} games each | {int(SIMS * 1.5)} sims/move | "
          f"inference: {'DirectML' if USE_GPU else 'CPU EP'} | "
          f"waiting: {'pure spin' if NO_EVENTS else f'spin({SPIN}) then event'}")
    q = mp.Queue()
    procs = [mp.Process(target=worker,
                        args=(i, shms[i].name, single_in, single_out, GAMES, SIMS, q,
                              req_ev, resp_evs[i], SPIN))
             for i in range(W)]
    t0 = time.perf_counter()
    for p in procs:
        p.start()
    rows = [q.get() for _ in range(W)]
    for p in procs:
        p.join(30)
    wall = time.perf_counter() - t0
    server.terminate()
    for s in shms:
        s.close()
        try:
            s.unlink()
        except Exception:
            pass

    print(f"\n{'wid':>4} {'wall s':>9} {'CPU s':>9} {'CPU/wall':>9} {'wait %':>8} "
          f"{'calls':>8} {'us/call':>9} {'timeouts':>9}")
    tot = send = recv = calls = tmo = wake = cputot = 0.0
    for wid, t, s, r, c, mv, to, wk, cp in sorted(rows):
        tot += t; send += s; recv += r; calls += c; tmo += to; wake += wk; cputot += cp
        print(f"{wid:>4} {t:>9.2f} {cp:>9.2f} {cp / t:>9.2f} {(s + r) / t * 100:>7.1f}% "
              f"{c:>8,} {r / max(1, c) * 1e6:>9.1f} {to:>9,}")
    print(f"{'-' * 78}")
    print(f"{'ALL':>4} {tot:>9.2f} {cputot:>9.2f} {cputot / tot:>9.2f} "
          f"{(send + recv) / tot * 100:>7.1f}% {int(calls):>8,}")
    print("\nCPU/wall is the headline: 1.00 = burning a full core throughout, "
          "lower = actually slept")
    print(f"total worker CPU burned: {cputot:.1f}s over {tot:.1f}s of worker wall time")
    if tmo + wake:
        print(f"     event waits: {tmo / (tmo + wake) * 100:.1f}% timed out "
              f"(each costs 50ms; high here means SPIN_LIMIT/timeout need tuning)")
    print(f"\nwall {wall:.1f}s | time spent waiting = {(send + recv) / tot * 100:.1f}% "
          f"(waiting is fine; burning a core while waiting is not - see CPU/wall)")
    print(f"  send wait (slot busy, i.e. contention) : {send / tot * 100:5.1f}%")
    print(f"  recv wait (inference latency)          : {recv / tot * 100:5.1f}%")
    print(f"  real search work                       : {(tot - send - recv) / tot * 100:5.1f}%")


if __name__ == "__main__":
    main()
