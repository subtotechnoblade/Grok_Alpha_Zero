"""Is the memo (Cache_Wrapper) slowing the search? Single worker, NO network.

arms:
  none  : session=None            -> MCTS dummy path, no input prep at all
  fake  : FakeSession             -> full request plumbing (input state, feed dict,
                                     .run call), zero eval cost
  memo  : Cache_Wrapper(Fake, 10) -> production memo config on top

memo overhead per request = (memo - fake) / requests.
plumbing overhead        = (fake - none).
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

from MCTS import MCTS
from Connect4.game import Connect4
from Session_Cache import Cache_Wrapper

SIMS = 375
GAMES = 5
NUM_EXPLORE = 3


class FakeSession:
    """Instant session: uniform policy, value 0. Shapes match onnxruntime."""
    def __init__(self):
        self.calls = 0
        self._policy = np.full((1, 7), 1.0 / 7.0, dtype=np.float32)
        self._value = np.zeros((1, 1), dtype=np.float32)

    def run(self, output_names, input_feed):
        self.calls += 1
        return self._policy, self._value


def play_one(session):
    game = Connect4()
    mcts = MCTS(game=game, session=session, use_njit=True, c_puct_init=1.5,
                use_dirichlet=True, dirichlet_alpha=1.33, dirichlet_epsilon=0.25,
                tau=1.0, fast_find_win=False)
    winner = -2
    while winner == -2 and len(game.action_history) < 42:
        cur = len(game.action_history)
        explore = (cur // 2 if cur % 2 == 0 else (cur - 1) // 2) < NUM_EXPLORE
        mcts.update_hyperparams(tau=1.0 if explore else 0)
        action, _ = mcts.run(iteration_limit=SIMS, time_limit=None, use_bar=False)
        game.do_action(action)
        winner = game.check_win()
        if winner == -2:
            mcts.prune_tree(action, create_new_root=True)
    return len(game.action_history)


def bench(label, make_session):
    np.random.seed(42)
    play_one(make_session())          # njit + code-path warmup, not timed
    times, moves, sess = [], 0, None
    np.random.seed(123)
    for _ in range(GAMES):
        sess = make_session()          # fresh session per game, like a fresh memo
        t0 = time.perf_counter()
        moves += play_one(sess)
        times.append(time.perf_counter() - t0)
    per = sum(times) / GAMES
    extra = ""
    if isinstance(sess, Cache_Wrapper):
        s = sess.stats()
        extra = (f"  last-game memo: {s['hits']}h/{s['misses']}m "
                 f"({s['hit_rate']:.0%}), {s['entries']} entries")
    elif isinstance(sess, FakeSession):
        extra = f"  last-game session calls: {sess.calls}"
    print(f"{label:<6} {per:7.2f} s/game   avg moves {moves / GAMES:5.1f}{extra}",
          flush=True)
    return per


def main():
    print(f"single worker, {SIMS} sims, dirichlet on, tau sched, fresh root, "
          f"{GAMES} games/arm (+1 warmup)\n")
    t_none = bench("none", lambda: None)
    t_fake = bench("fake", lambda: FakeSession())
    t_memo = bench("memo", lambda: Cache_Wrapper(FakeSession(), 10))
    print()
    print(f"plumbing cost (fake - none): {t_fake - t_none:+.2f} s/game")
    print(f"memo cost (memo - fake)    : {t_memo - t_fake:+.2f} s/game")


if __name__ == "__main__":
    main()
