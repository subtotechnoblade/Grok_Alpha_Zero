"""MCTS smoke for every game: scalar actions (Connect4) and 2-D actions (Gomoku)."""
import os
import sys
import traceback

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from MCTS import MCTS

GAMES = sys.argv[1:] or ["Connect4", "Gomoku", "TicTacToe"]

for name in GAMES:
    try:
        mod = __import__(f"{name}.game", fromlist=[name])
        game = getattr(mod, name)()
        mcts = MCTS(game, None, fast_find_win=False)
        n = len(game.get_legal_actions())
        action, probs = mcts.run(iteration_limit=n, use_bar=False)
        hist = game.action_history
        print(f"{name:<10} OK   legal={n:<4} action={action}  "
              f"action_ndim={getattr(action, 'ndim', 0)}")
    except Exception:
        print(f"{name:<10} RAISED:")
        traceback.print_exc()
