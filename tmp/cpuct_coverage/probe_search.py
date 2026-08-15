"""CONTROLLED tactical probe: the SAME exactly-solved positions are handed to the search
at every c_puct value, so only the search differs. Measures how often the search picks a
game-theoretically optimal column.

Only positions where the choice actually matters are used (>1 distinct outcome among the
legal moves), otherwise every move is "optimal" and the metric is vacuous.

Search settings match self play at a tau=0 ply: 375 sims, no dirichlet, fresh root.

usage: probe_search.py <solved.npz> [n] [workers] [model] [sims]
"""
import sys
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero")
MODEL = str(ROOT / "tmp/run3_archive_gen0-15_bottleneck_preweightfix/12/model.onnx")
CPUCTS = [1.5, 2.5, 3.5, 5.0]
_SESS = None


def _one(args):
    board, mover, hist, signs, cp, model, sims, seed = args
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import onnxruntime as rt
    rt.set_default_logger_severity(4)
    from MCTS import MCTS
    from Connect4.game import Connect4

    global _SESS
    if _SESS is None:
        so = rt.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        _SESS = rt.InferenceSession(model, so, providers=["CPUExecutionProvider"])
    np.random.seed(seed)

    g = Connect4()
    g.board = board.reshape(6, 7).astype(np.int8).copy()
    g.next_player = int(mover)
    g.action_history = [int(x) for x in hist if x >= 0]   # the REAL history, so the
    # 2-ply history planes of the net input are exactly what self play would have fed

    tree = MCTS(game=g, session=_SESS, use_njit=True, c_puct_init=cp,
                use_dirichlet=False, tau=0, fast_find_win=False)
    action, _ = tree.run(iteration_limit=sims, time_limit=None, use_bar=False)
    a = int(action)
    best = signs[signs != 999].max()
    return cp, int(signs[a] == best), int(signs[a]), int(best)


def main():
    solved = np.load(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    model = sys.argv[4] if len(sys.argv) > 4 else MODEL
    sims = int(sys.argv[5]) if len(sys.argv) > 5 else 375

    boards, movers, signs, ok, plies = (solved["boards"], solved["movers"],
                                        solved["signs"], solved["ok"], solved["plies"])
    keep = ok == 1
    nontrivial = np.array([len(set(s[s != 999].tolist())) > 1 for s in signs])
    keep &= nontrivial
    idx = np.where(keep)[0]
    rng = np.random.default_rng(1)
    idx = rng.permutation(idx)[:n]
    print(f"{len(signs)} solved positions -> {nontrivial.sum()} non-trivial -> "
          f"probing {len(idx)}", flush=True)
    print(f"ply distribution of probe set: min {plies[idx].min()} "
          f"median {np.median(plies[idx]):.0f} max {plies[idx].max()}", flush=True)

    jobs = [(boards[i], movers[i], solved["hists"][i], signs[i], cp, model, sims,
             5000 + int(i)) for i in idx for cp in CPUCTS]
    t0 = time.time()
    with mp.Pool(workers) as pool:
        res = list(pool.imap_unordered(_one, jobs, chunksize=8))
    print(f"{len(jobs)} searches in {time.time() - t0:.0f}s\n")

    print(f"{'c_puct':>7} {'n':>6} {'optimal %':>10} {'+-':>6} "
          f"{'lost a WIN %':>13} {'threw a DRAW %':>15}")
    for cp in CPUCTS:
        r = [x for x in res if x[0] == cp]
        opt = np.array([x[1] for x in r])
        got = np.array([x[2] for x in r])
        best = np.array([x[3] for x in r])
        lostwin = ((best == 1) & (got < 1)).sum() / max((best == 1).sum(), 1)
        threwdraw = ((best == 0) & (got < 0)).sum() / max((best == 0).sum(), 1)
        se = (opt.mean() * (1 - opt.mean()) / len(opt)) ** 0.5
        print(f"{cp:>7} {len(opt):>6} {100 * opt.mean():>10.1f} {100 * se:>6.1f} "
              f"{100 * lostwin:>13.1f} {100 * threwdraw:>15.1f}")


if __name__ == "__main__":
    main()
