"""Head-to-head match at the PRODUCTION sim count. No reference search, so nothing
assumes a deeper search of the same net is ground truth -- the only thing measured is
which rule actually wins games.

Design:
  * every one of the 49 two-ply openings is played TWICE, colours swapped, so opening
    choice and first-player advantage cancel exactly. Paired, not sampled.
  * deterministic after the opening (tau=0, no dirichlet) so a pair differs ONLY in
    which side used which rule.
  * fresh root each move, matching create_new_root=True in self play.

  usage: fpu_match.py [n_openings] [workers]
"""
import os
import sys
import time
import itertools
import multiprocessing as mp

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL = "Connect4/Grok_Zero_Train/6/model.onnx"
# At 375 the initialisation governs ~2.7% of the budget (7 children all visited within
# the first ~10 sims). Drop the budget toward the branching factor and that fraction
# approaches 100% -- this is the regime where FPU should bite if it ever does.
SIMS = int(os.environ.get("FPU_SIMS", "375"))
REDUCTION = float(os.environ.get("FPU_REDUCTION", "0.0"))  # 0 = unvisited child = parent Q


def _setup():
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    import numpy as np
    import onnxruntime as rt
    rt.set_default_logger_severity(4)
    from MCTS import MCTS
    from Connect4.game import Connect4
    return np, rt, MCTS, Connect4


def play_game(args):
    opening, fpu_first = args
    np, rt, MCTS, Connect4 = _setup()

    original = MCTS._get_best_PUCT_score_index

    def puct_fpu(child_prob_priors, child_values, child_visits, parent_visits,
                 c_puct_init, c_puct_base):
        U = child_prob_priors * ((parent_visits ** 0.5) / (child_visits + 1)) * (
            c_puct_init + np.log((parent_visits + c_puct_base + 1) / c_puct_base))
        mask = child_visits > 0
        total = child_visits.sum()
        pq = (child_values.sum() / total) if total > 0 else 0.0
        Q = np.full(child_values.shape, pq - REDUCTION, dtype=np.float32)
        Q[mask] = child_values[mask] / child_visits[mask]
        return int(np.argmax(Q + U))

    sess = rt.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    game = Connect4()
    for a in opening:
        game.do_action(int(a))
    if game.check_win() != -2:
        return None

    # player -1 moves first. fpu_first says whether FPU is playing -1 this game.
    while True:
        fpu_to_move = (game.next_player == -1) == fpu_first
        MCTS._get_best_PUCT_score_index = staticmethod(
            puct_fpu if fpu_to_move else original)
        try:
            tree = MCTS(game=game, session=sess, use_njit=False, c_puct_init=1.5,
                        use_dirichlet=False, tau=0, fast_find_win=False)
            action, _ = tree.run(iteration_limit=SIMS, time_limit=None, use_bar=False)
        finally:
            MCTS._get_best_PUCT_score_index = staticmethod(original)
        game.do_action(int(action))
        w = game.check_win()
        if w != -2:
            break
        if len(game.action_history) >= 42:
            w = 0
            break
    # translate to FPU's perspective: +1 FPU won, -1 FPU lost, 0 draw
    if w == 0:
        return 0
    fpu_player = -1 if fpu_first else 1
    return 1 if w == fpu_player else -1


def main():
    n_open = int(sys.argv[1]) if len(sys.argv) > 1 else 49
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    # 3-ply openings (343 combos) give far more distinct starting points than the 49
    # two-ply ones. Shuffled with a fixed seed so a truncated run is not biased toward
    # low column indices.
    import random
    ply = int(os.environ.get("OPENING_PLY", "3"))
    allo = list(itertools.product(range(7), repeat=ply))
    random.Random(0).shuffle(allo)
    openings = allo[:n_open]
    jobs = [(o, first) for o in openings for first in (True, False)]

    print(f"net run3 gen 6 | {SIMS} sims | FPU = parent Q - {REDUCTION}")
    print(f"{len(openings)} openings x 2 colours = {len(jobs)} games, {workers} workers")
    t0 = time.time()
    with mp.Pool(workers) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(play_game, jobs), 1):
            res.append(r)
            if i % 10 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(jobs)} games, {el:.0f}s elapsed, "
                      f"eta {el / i * (len(jobs) - i):.0f}s", flush=True)

    res = [r for r in res if r is not None]
    w = sum(1 for r in res if r == 1)
    l = sum(1 for r in res if r == -1)
    d = sum(1 for r in res if r == 0)
    n = len(res)
    score = (w + 0.5 * d) / n
    se = (0.25 / n) ** 0.5
    print(f"\n=== {n} games, {time.time() - t0:.0f}s ===")
    print(f"FPU wins {w}   losses {l}   draws {d}")
    print(f"FPU score {score:.3f} +/- {se:.3f}   (0.500 = no difference)")
    # decisive-games sign test
    from math import comb
    nd = w + l
    k = min(w, l)
    p = (min(1.0, 2 * sum(comb(nd, i) for i in range(k + 1)) / (2 ** nd))
         if nd else 1.0)
    print(f"sign test on {nd} decisive games: exact two-sided p = {p:.3f}"
          f"  -> {'SIGNIFICANT' if p < 0.05 else 'not significant'}")


if __name__ == "__main__":
    main()
