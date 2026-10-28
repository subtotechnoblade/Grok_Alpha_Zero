"""How close each search setting's root visit distribution comes to a big K=1 reference search.
python -m tests.policy_target_quality <model_dir> [positions] [workers] [reference_sims]
Positions come from quick self-play games (tau 1, Dirichlet on); every search starts from a fresh tree without noise."""
import multiprocessing as mp
import sys
import time

import numpy as np

from tests.mcts_scenarios import ROOT
from tests.bench_search_speed import providers_for, model_path
from Gomoku.game import Gomoku

SETTINGS = [(1, 600), (1, 1800), (8, 1800)]
K_MAX = 8
EPS = 1e-6


def search(cached, history, k, sims):
    from MCTS import MCTS
    game = Gomoku()
    for move in history:
        game.do_action(move)
    mcts = MCTS(game, cached, c_puct_init=2.0, fpu_reduction=0.2, use_dirichlet=False, tau=0.0,
                leaves_per_batch=k)
    mcts.run(iteration_limit=sims, use_bar=False)
    visits = np.zeros(225, np.float64)
    root = mcts.root
    for i, child in enumerate(root.children):
        x, y = child.action_history[-1]
        visits[int(y) * 15 + int(x)] = root.child_visits[i]
    return visits / visits.sum()


def metrics(ref, p):
    with np.errstate(all="ignore"):
        q = (p + EPS) / (1 + 225 * EPS)
        mask = ref > 0
        kl = float(np.sum(ref[mask] * (np.log(ref[mask]) - np.log(q[mask]))))
        tv = float(0.5 * np.abs(p - ref).sum())
        entropy = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
        top_agree = float(np.argmax(p) == np.argmax(ref))
        on_ref_best = float(p[np.argmax(ref)])
    return kl, tv, top_agree, on_ref_best, entropy


def sample_positions(cached, n, rng):
    from MCTS import MCTS
    positions = []
    while len(positions) < n:
        game = Gomoku()
        mcts = MCTS(game, cached, c_puct_init=2.0, fpu_reduction=0.2, use_dirichlet=True, dirichlet_alpha=0.05,
                    tau=1.0, leaves_per_batch=K_MAX)
        plies = set(rng.choice(np.arange(0, 40), size=2, replace=False).tolist())
        while game.check_win() == -2 if game.action_history else True:
            if len(game.action_history) in plies:
                positions.append([tuple(int(c) for c in a) for a in game.action_history])
                if len(positions) == n:
                    break
            mcts.update_hyperparams(tau=1.0 if len(game.action_history) < 8 else 0.0)
            move, _ = mcts.run(iteration_limit=200, use_bar=False)
            game.do_action(move)
            if game.check_win() == -2:
                mcts.prune_tree(move)
    return positions


def worker(spec, shm, request_ready, response_ready, n, ref_sims, seed, results):
    from Client_Server import Batched_Session
    from Session_Cache import Cache_Wrapper
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    session = Batched_Session(spec, shm, request_ready, response_ready)
    cached = Cache_Wrapper(session, 3)
    for history in sample_positions(cached, n, rng):
        ref = search(cached, history, 1, ref_sims)
        row = {"ply": len(history), "ref_entropy": metrics(ref, ref)[4]}
        for k, sims in SETTINGS:
            row[(k, sims)] = metrics(ref, search(cached, history, k, sims))
        results.put(row)
    results.put(None)
    session.client.close()
    shm.close()


if __name__ == "__main__":
    model_dir = sys.argv[1]
    total = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    ref_sims = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
    from Client_Server import start_batch_server, create_wakeups
    from Transport import ChannelSpec, create_slots
    game = Gomoku()
    state = game.get_input_state()
    pmax = workers * K_MAX
    wire = state.dtype if state.dtype.kind in "biu" else np.float32
    spec = ChannelSpec({"inputs": (state.shape, wire)},
                       {"policy": (game.policy_shape, np.float32), "value": ((1,), np.float32)}, max_rows=K_MAX)
    shms = create_slots(spec, workers)
    request_ready, response_events = create_wakeups(workers)
    server = mp.Process(target=start_batch_server,
                        args=(spec, shms, providers_for(model_dir, state.shape, pmax), model_path(model_dir, pmax),
                              request_ready, response_events))
    server.start()
    results = mp.Queue()
    shares = [total // workers + (i < total % workers) for i in range(workers)]
    procs = [mp.Process(target=worker, args=(spec, shms[i], request_ready, response_events[i], shares[i], ref_sims,
                                              1000 + i, results)) for i in range(workers) if shares[i]]
    for p in procs:
        p.start()
    t0 = time.perf_counter()
    rows, finished = [], 0
    while finished < len(procs):
        row = results.get()
        if row is None:
            finished += 1
            continue
        rows.append(row)
        if len(rows) % 50 == 0:
            print(f"{len(rows)} positions ({time.perf_counter() - t0:.0f}s)", flush=True)
    for p in procs:
        p.join()
    server.terminate()
    server.join()
    for shm in shms:
        shm.close()
        shm.unlink()
    print(f"RESULT {len(rows)} positions (plies {min(r['ply'] for r in rows)}-{max(r['ply'] for r in rows)}), "
          f"reference K=1 {ref_sims} sims, entropy {np.mean([r['ref_entropy'] for r in rows]):.3f}")
    n = len(rows)
    names = ("KL(ref||p)", "TV distance", "top-1 agree", "mass on ref best", "entropy")
    print(f"{'setting':>12} " + " ".join(f"{name:>17}" for name in names))
    for k, sims in SETTINGS:
        m = np.array([r[(k, sims)] for r in rows])
        se = m.std(axis=0, ddof=1) / np.sqrt(n)
        print(f"{f'K={k} {sims}':>12} " + " ".join(f"{m[:, j].mean():9.3f}+-{1.96 * se[j]:.3f}" for j in range(5)))
    for j, name in enumerate(names[:4]):
        a = np.array([r[(8, 1800)][j] for r in rows])
        for k, sims in [(1, 600), (1, 1800)]:
            d = a - np.array([r[(k, sims)][j] for r in rows])
            print(f"paired {name}, K=8 1800 minus K={k} {sims}: {d.mean():+.4f} +- {1.96 * d.std(ddof=1) / np.sqrt(n):.4f}")
