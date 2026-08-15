"""Paired head-to-head match between two c_puct_init values, same fixed net.

Same design as tmp/fpu_match.py: every opening is played TWICE with colours swapped,
deterministic after the opening (tau=0, no dirichlet), fresh root each move
(create_new_root=True in self play). A pair differs ONLY in which side used which c_puct.

usage: cpuct_match.py <c_puct_test> <c_puct_base> <n_openings> <workers> [model] [sims]
"""
import os
import sys
import time
import random
import itertools
import multiprocessing as mp

ROOT = r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero"
MODEL = os.path.join(ROOT, "tmp/run3_archive_gen0-15_bottleneck_preweightfix/12/model.onnx")
SIMS = 375
OPENING_PLY = 3

_SESS = None


def _sess(model):
    global _SESS
    if _SESS is None:
        sys.path.insert(0, ROOT)
        import onnxruntime as rt
        rt.set_default_logger_severity(4)
        so = rt.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        _SESS = rt.InferenceSession(model, so, providers=["CPUExecutionProvider"])
    return _SESS


def play_game(args):
    opening, test_first, cp_test, cp_base, model, sims = args
    sys.path.insert(0, ROOT)
    import numpy as np
    from MCTS import MCTS
    from Connect4.game import Connect4

    sess = _sess(model)
    game = Connect4()
    for a in opening:
        game.do_action(int(a))
        if game.check_win() != -2:
            return None

    while True:
        test_to_move = (game.next_player == -1) == test_first
        cp = cp_test if test_to_move else cp_base
        tree = MCTS(game=game, session=sess, use_njit=True, c_puct_init=cp,
                    use_dirichlet=False, tau=0, fast_find_win=False)
        action, _ = tree.run(iteration_limit=sims, time_limit=None, use_bar=False)
        game.do_action(int(action))
        w = game.check_win()
        if w != -2:
            break
        if len(game.action_history) >= 42:
            w = 0
            break
    if w == 0:
        return 0
    test_player = -1 if test_first else 1
    return 1 if w == test_player else -1


def main():
    cp_test = float(sys.argv[1])
    cp_base = float(sys.argv[2])
    n_open = int(sys.argv[3])
    workers = int(sys.argv[4])
    model = sys.argv[5] if len(sys.argv) > 5 else MODEL
    sims = int(sys.argv[6]) if len(sys.argv) > 6 else SIMS

    allo = list(itertools.product(range(7), repeat=OPENING_PLY))
    random.Random(0).shuffle(allo)
    openings = allo[:n_open]
    jobs = [(o, first, cp_test, cp_base, model, sims)
            for o in openings for first in (True, False)]

    print(f"c_puct {cp_test} (test) vs {cp_base} (base) | {sims} sims | {model}")
    print(f"{len(openings)} openings x 2 colours = {len(jobs)} games, {workers} workers",
          flush=True)
    t0 = time.time()
    with mp.Pool(workers) as pool:
        res = [r for r in pool.imap_unordered(play_game, jobs)]

    res = [r for r in res if r is not None]
    w = sum(1 for r in res if r == 1)
    l = sum(1 for r in res if r == -1)
    d = sum(1 for r in res if r == 0)
    n = len(res)
    score = (w + 0.5 * d) / n
    se = (0.25 / n) ** 0.5
    print(f"\n=== {n} games, {time.time() - t0:.0f}s ===")
    print(f"c_puct={cp_test} wins {w}  losses {l}  draws {d}")
    print(f"score {score:.3f} +/- {se:.3f}  (0.500 = no difference)")
    from math import comb
    nd = w + l
    k = min(w, l)
    p = (min(1.0, 2 * sum(comb(nd, i) for i in range(k + 1)) / (2 ** nd)) if nd else 1.0)
    print(f"sign test on {nd} decisive games: exact two-sided p = {p:.3f} -> "
          f"{'SIGNIFICANT' if p < 0.05 else 'not significant'}", flush=True)


if __name__ == "__main__":
    main()
