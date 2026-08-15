"""Replicate Self_Play.play() move-for-move with the gen-9 net on CPU and measure:
- distinct games out of N
- where randomness enters (tau plies vs argmax plies)
- ply-0 root visit distribution across games (does dirichlet move it?)
- inherited vs fresh visits at each re-rooted root (tree reuse bias)
- assert tau=0 chooses argmax(visits)

Usage: sp_experiment.py N_GAMES CREATE_NEW_ROOT USE_DIRICHLET
"""
import sys, time
import numpy as np
import onnxruntime as rt

sys.path.insert(0, ".")
from MCTS import MCTS
from Connect4.game import Connect4
from Session_Cache import Cache_Wrapper

N_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 20
CREATE_NEW_ROOT = bool(int(sys.argv[2])) if len(sys.argv) > 2 else False
USE_DIRICHLET = bool(int(sys.argv[3])) if len(sys.argv) > 3 else True

ITERS = int(250 * 1.5)          # Self_Play: int(iteration_limit * 1.5)
N_EXPL_FIRST = 3                # train_config values
N_EXPL_SECOND = 3
MAX_ACTIONS = 42

so = rt.SessionOptions()
so.intra_op_num_threads = 2
so.log_severity_level = 3
raw = rt.InferenceSession("Connect4/Grok_Zero_Train/9/model.onnx",
                          sess_options=so, providers=["CPUExecutionProvider"])
# memo across games: outputs are a pure function of the input, so this cannot
# change behaviour, only speed (same class Self_Play uses, wider depth window)
session = Cache_Wrapper(raw, max_cache_depth=100, max_entries=2_000_000)

games = []
ply0_dists = []
inherited_stats = []            # (ply, inherited_visits, total_after_run)
argmax_violations = 0

t0 = time.time()
for g in range(N_GAMES):
    game = Connect4()
    mcts = MCTS(game=game, session=session, use_njit=None,
                c_puct_init=1.5, use_dirichlet=USE_DIRICHLET,
                dirichlet_alpha=1.33, dirichlet_epsilon=0.25,
                tau=1.0, fast_find_win=False)
    seq = []
    winner = -2
    count = 0
    while winner == -2 and count < MAX_ACTIONS:
        ply = len(game.action_history)
        if ply % 2 == 0:
            explore = ply // 2 < N_EXPL_FIRST
        else:
            explore = (ply + 1) // 2 < N_EXPL_SECOND
        mcts.update_hyperparams(tau=1.0 if explore else 0)

        inherited = mcts.root.visits  # visits already on the root BEFORE this move's sims
        action, action_probs = mcts.run(iteration_limit=ITERS, time_limit=None, use_bar=False)
        total = mcts.root.visits
        inherited_stats.append((ply, inherited, total))

        if ply == 0:
            # child order is sorted by noised prior; map action -> visits
            d = {int(c.action_history[-1]): v
                 for c, v in zip(mcts.root.children, mcts.root.child_visits)}
            ply0_dists.append([d.get(a, 0) for a in range(7)])
        if not explore:
            best = int(np.argmax(mcts.root.child_visits))
            if mcts.root.children[best].action_history[-1] != action:
                argmax_violations += 1

        seq.append(int(action))
        game.do_action(action)
        winner = game.check_win()
        if winner == -2:
            mcts.prune_tree(action, CREATE_NEW_ROOT)
        count += 1
        if count == MAX_ACTIONS:
            winner = 0
    games.append(tuple(seq))

el = time.time() - t0
print(f"config: create_new_root={CREATE_NEW_ROOT} use_dirichlet={USE_DIRICHLET} "
      f"games={N_GAMES} iters/move={ITERS}  ({el:.0f}s, memo {session.stats()['hit_rate']:.1%} hits)")
print(f"distinct games: {len(set(games))} / {N_GAMES}")
from collections import Counter
fm = Counter(s[0] for s in games)
print(f"first moves: {dict(sorted(fm.items()))}")
# distinct prefixes
for p in (2, 4, 6, 8, 12, 20):
    pref = set(s[:p] for s in games if len(s) >= p)
    print(f"  distinct prefixes ply {p:2d}: {len(pref)}")
print(f"tau=0 argmax violations: {argmax_violations}")

ply0 = np.array(ply0_dists, dtype=np.float64)
ply0 = ply0 / ply0.sum(axis=1, keepdims=True)
print(f"ply-0 visit distribution: mean {np.array2string(ply0.mean(0), precision=3, suppress_small=True)}")
print(f"                          std  {np.array2string(ply0.std(0), precision=3, suppress_small=True)}")
print(f"ply-0 argmax column histogram: {dict(sorted(Counter(np.argmax(ply0, 1).tolist()).items()))}")

# inherited fraction by ply bucket
arr = np.array(inherited_stats, dtype=np.float64)
print("inherited root visits (tree reuse) by ply:")
for lo, hi in ((1, 5), (5, 10), (10, 20), (20, 42)):
    m = (arr[:, 0] >= lo) & (arr[:, 0] < hi)
    if m.any():
        inh, tot = arr[m, 1], arr[m, 2]
        print(f"  plies {lo:2d}-{hi:2d}: mean inherited {inh.mean():6.0f}, "
              f"mean total after run {tot.mean():6.0f}, inherited fraction {np.mean(inh / tot):.2f}")
print()
print("example games:")
for s in games[:6]:
    print(" ", s)
