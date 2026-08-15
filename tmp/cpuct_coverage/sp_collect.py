"""Self-play position collector, faithful to Self_Play.Self_Play.play() except that
c_puct_init is a parameter and nothing is written to any h5 / no training happens.

Production settings copied from Connect4/config.py + Self_Play.py:
  sims = int(250 * 1.5) = 375, use_dirichlet=True, alpha=1.33, eps=0.25,
  tau schedule 3 explore plies per player, create_new_root=True, fast_find_win=False.

Per ply it records:
  game id, ply, board-position hash, net-input-state hash (includes 2 ply of history),
  root visit entropy -> effective branching, top-1 visit share, PV depth,
  tactical ground truth flags (immediate win available / taken, forced block / taken).

usage: sp_collect.py <c_puct> <n_games> <seed> <out.npz> [model] [sims]
"""
import sys
import time
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero")
sys.path.insert(0, str(ROOT))

import onnxruntime as rt

rt.set_default_logger_severity(4)

from Connect4.game import Connect4
from MCTS import MCTS

NUM_EXPLORE_FIRST = 3
NUM_EXPLORE_SECOND = 3
MAX_ACTIONS = 42


def h64(b: bytes) -> np.uint64:
    return np.frombuffer(hashlib.blake2b(b, digest_size=8).digest(), dtype=np.uint64)[0]


def wins_now(board, action, player):
    """True if `player` playing `action` on `board` wins immediately."""
    nb = Connect4.do_action_MCTS(board.copy(), action, player)
    hist = np.array([action], dtype=np.int64)  # check_win only uses hist[-1]
    return Connect4.check_win_MCTS(nb, player, hist) == player


def legal(board):
    return [x for x in range(7) if np.abs(board[:, x]).sum() < 6]


def tactics(board, player):
    """Ground truth 1-ply tactics for `player` to move.
    returns (win_moves, safe_moves, n_legal)
      win_moves : moves that win on the spot
      safe_moves: moves after which the opponent has NO immediate winning reply
    """
    ls = legal(board)
    wins = [a for a in ls if wins_now(board, a, player)]
    safe = []
    for a in ls:
        nb = Connect4.do_action_MCTS(board.copy(), a, player)
        opp_ls = legal(nb)
        if not any(wins_now(nb, b, -player) for b in opp_ls):
            safe.append(a)
    return wins, safe, len(ls)


def pv_depth(root):
    """Length of the principal variation: follow max-visit child while children exist."""
    node = root
    d = 0
    while True:
        kids = getattr(node, "children", None)
        vis = getattr(node, "child_visits", None)
        if not kids or vis is None or len(vis) == 0 or max(vis) == 0:
            return d
        i = int(np.argmax(np.asarray(vis[:len(kids)])))
        node = kids[i]
        d += 1
        if d > 45:
            return d


def play_one(sess, c_puct, sims, gid, rows, boards, movers, hists):
    game = Connect4()
    mcts = MCTS(game=game, session=sess, use_njit=True,
                c_puct_init=c_puct, use_dirichlet=True,
                dirichlet_alpha=1.33, dirichlet_epsilon=0.25,
                tau=1.0, fast_find_win=False)
    winner = -2
    n = 0
    while winner == -2 and n < MAX_ACTIONS:
        ply = len(game.action_history)
        mover = game.next_player
        board = game.board.copy()

        state_key = h64(game.get_input_state().tobytes())
        board_key = h64(board.tobytes() + bytes([mover & 0xFF]))

        if ply % 2 == 0:
            explore = ply // 2 < NUM_EXPLORE_FIRST
        else:
            explore = (ply - 1) // 2 < NUM_EXPLORE_SECOND
        mcts.update_hyperparams(tau=1.0 if explore else 0)

        wins, safe, n_legal = tactics(board, mover)

        action, move_probs = mcts.run(iteration_limit=sims, time_limit=None, use_bar=False)

        v = np.asarray(mcts.root.child_visits, dtype=np.float64)
        tot = v.sum()
        p = v / tot if tot > 0 else np.ones_like(v) / len(v)
        nz = p[p > 0]
        ent = float(-(nz * np.log(nz)).sum())
        eff_b = float(np.exp(ent))
        top1 = float(p.max())
        pvd = pv_depth(mcts.root)

        a = int(action)
        rows.append((gid, ply, board_key, state_key, eff_b, top1, pvd,
                     int(explore), n_legal,
                     len(wins), int(a in wins),
                     len(safe), int(a in safe)))
        boards.append(board.reshape(-1))
        movers.append((mover, a))
        hh = np.full(MAX_ACTIONS, -1, dtype=np.int8)
        if ply:
            hh[:ply] = np.asarray(game.action_history, dtype=np.int8)
        hists.append(hh)

        game.do_action(action)
        winner = game.check_win()
        if winner == -2:
            mcts.prune_tree(action, True)  # create_new_root=True
        n += 1
        if n == MAX_ACTIONS and winner == -2:
            winner = 0
    if winner == -2:
        winner = 0
    return int(winner), len(game.action_history)


def main():
    c_puct = float(sys.argv[1])
    n_games = int(sys.argv[2])
    seed = int(sys.argv[3])
    out = sys.argv[4]
    model = sys.argv[5] if len(sys.argv) > 5 else str(
        ROOT / "tmp/run3_archive_gen0-15_bottleneck_preweightfix/12/model.onnx")
    sims = int(sys.argv[6]) if len(sys.argv) > 6 else 375

    np.random.seed(seed)
    so = rt.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = rt.InferenceSession(model, so, providers=["CPUExecutionProvider"])

    rows = []
    games = []
    boards = []
    movers = []
    hists = []
    t0 = time.perf_counter()
    for g in range(n_games):
        w, l = play_one(sess, c_puct, sims, g, rows, boards, movers, hists)
        games.append((g, w, l))
    wall = time.perf_counter() - t0

    cols = np.array([r[:2] for r in rows], dtype=np.int64)
    keys = np.array([(r[2], r[3]) for r in rows], dtype=np.uint64)
    fl = np.array([(r[4], r[5]) for r in rows], dtype=np.float32)
    it = np.array([r[6:] for r in rows], dtype=np.int32)
    np.savez_compressed(out, gid_ply=cols, keys=keys, floats=fl, ints=it,
                        games=np.array(games, dtype=np.int64),
                        boards=np.array(boards, dtype=np.int8),
                        movers=np.array(movers, dtype=np.int8),
                        hists=np.array(hists, dtype=np.int8),
                        meta=np.array([c_puct, sims, seed, n_games, wall], dtype=np.float64))
    print(f"c_puct={c_puct} n={n_games} seed={seed} sims={sims} "
          f"wall={wall:.1f}s ({wall / n_games:.2f}s/game) positions={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
