"""Strength match on Gomoku through the batched server between two search settings, colors alternating.
python -m tests.match_virtual_loss <model_dir> [games] [A] [B] [workers]
A and B are K:virtual_loss:sims, e.g. 8:1.0:1800 vs 1:1.0:600 (the defaults). Scores are A's.
The first OPENING_PLIES moves are sampled at tau 1 so games differ, then both sides play their most visited move."""
import math
import multiprocessing as mp
import sys
import time

import numpy as np

from tests.mcts_scenarios import ROOT
from tests.bench_search_speed import providers_for, model_path
from Gomoku.game import Gomoku

OPENING_PLIES = 4
SIDE_STATS = ("batches", "network_rows", "collisions", "sims", "seconds")


def parse_side(text):
    k, vl, sims = text.split(":")
    return int(k), float(vl), int(sims)


def play_games(spec, shm, request_ready, response_ready, game_ids, a, b, results):
    from Client_Server import Batched_Session
    from MCTS import MCTS
    from Session_Cache import Cache_Wrapper
    np.random.seed()
    session = Batched_Session(spec, shm, request_ready, response_ready)
    cached = Cache_Wrapper(session, 3)
    for game_id in game_ids:
        game = Gomoku()
        hits_before = cached.hits
        a_side = -1 if game_id % 2 == 0 else 1
        players, budget, stats = {}, {}, {}
        for side in (-1, 1):
            k, vl, sims = a if side == a_side else b
            budget[side] = sims
            stats[side] = dict.fromkeys(SIDE_STATS, 0.0)
            players[side] = MCTS(game, cached, c_puct_init=2.0, fpu_reduction=0.2, use_dirichlet=False,
                                 tau=1.0, leaves_per_batch=k, virtual_loss=vl)
        winner = -2
        while winner == -2:
            mover = game.get_next_player()
            mcts = players[mover]
            mcts.update_hyperparams(tau=1.0 if len(game.action_history) < OPENING_PLIES else 0.0)
            t0 = time.perf_counter()
            move, _ = mcts.run(iteration_limit=budget[mover], use_bar=False)
            stats[mover]["seconds"] += time.perf_counter() - t0
            stats[mover]["sims"] += budget[mover]
            game.do_action(move)
            winner = game.check_win()
            if winner == -2:
                for player in players.values():
                    player.prune_tree(move)
        for side, player in players.items():
            for key in ("batches", "network_rows", "collisions"):
                stats[side][key] = player.search_stats[key]
        score = 0.5 if winner == 0 else float(winner == a_side)
        evaluated = sum(p.search_stats["network_rows"] for p in players.values()) - (cached.hits - hits_before)
        results.put((game_id, a_side == -1, score, len(game.action_history), evaluated,
                     [stats[a_side][key] for key in SIDE_STATS], [stats[-a_side][key] for key in SIDE_STATS]))
    session.client.close()
    shm.close()


def describe(name, setting, totals):
    batches, rows, collisions, sims, seconds = totals
    return (f"{name} K={setting[0]} vl={setting[1]} sims={setting[2]}: {rows / max(batches, 1):.2f} rows/batch, "
            f"{collisions / max(sims, 1):.3f} collisions/sim, {1e3 * seconds / max(sims, 1):.3f} ms/sim")


if __name__ == "__main__":
    model_dir = sys.argv[1]
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    a = parse_side(sys.argv[3]) if len(sys.argv) > 3 else (8, 1.0, 1800)
    b = parse_side(sys.argv[4]) if len(sys.argv) > 4 else (1, 1.0, 600)
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 12
    from Client_Server import start_batch_server, create_wakeups
    from Transport import ChannelSpec, create_slots
    game = Gomoku()
    state = game.get_input_state()
    k = max(a[0], b[0])
    pmax = workers * k
    wire = state.dtype if state.dtype.kind in "biu" else np.float32
    spec = ChannelSpec({"inputs": (state.shape, wire)},
                       {"policy": (game.policy_shape, np.float32), "value": ((1,), np.float32)}, max_rows=k)
    shms = create_slots(spec, workers)
    request_ready, response_events = create_wakeups(workers)
    server = mp.Process(target=start_batch_server,
                        args=(spec, shms, providers_for(model_dir, state.shape, pmax), model_path(model_dir, pmax),
                              request_ready, response_events))
    server.start()
    results = mp.Queue()
    procs = [mp.Process(target=play_games, args=(spec, shms[i], request_ready, response_events[i],
                                                  list(range(i, games, workers)), a, b, results))
             for i in range(workers)]
    for p in procs:
        p.start()
    t0 = time.perf_counter()
    rows = []
    for n in range(games):
        rows.append(results.get())
        if (n + 1) % 50 == 0:
            s = np.array([r[2] for r in rows])
            elapsed = time.perf_counter() - t0
            print(f"{n + 1} games: A scores {s.mean():.3f} ({elapsed:.0f}s, "
                  f"{sum(r[4] for r in rows) / elapsed:,.0f} evaluated positions/s)", flush=True)
    for p in procs:
        p.join()
    elapsed = time.perf_counter() - t0
    server.terminate()
    server.join()
    for shm in shms:
        shm.close()
        shm.unlink()
    s = np.array([r[2] for r in rows])
    first = np.array([r[2] for r in rows if r[1]])
    second = np.array([r[2] for r in rows if not r[1]])
    se = s.std(ddof=1) / math.sqrt(len(s))
    elo = -400 * math.log10(1 / min(max(s.mean(), 1e-3), 1 - 1e-3) - 1)
    print(f"RESULT A {a} vs B {b}, {len(s)} games: A scores {s.mean():.3f} +- {1.96 * se:.3f} (95%), "
          f"elo {elo:+.0f}; as first {first.mean():.3f} ({len(first)}), as second {second.mean():.3f} "
          f"({len(second)}); wins {(s == 1).sum()} draws {(s == 0.5).sum()} losses {(s == 0).sum()}; "
          f"mean length {np.mean([r[3] for r in rows]):.1f}; "
          f"{sum(r[4] for r in rows) / elapsed:,.0f} evaluated positions/s over {elapsed:.0f}s", flush=True)
    print(describe("A", a, np.sum([r[5] for r in rows], axis=0)), flush=True)
    print(describe("B", b, np.sum([r[6] for r in rows], axis=0)), flush=True)
