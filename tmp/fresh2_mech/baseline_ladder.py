"""Baseline ladder: how does the first-player edge change as play gets more tactical?

L0: uniform random legal play
L1: take immediate win, else random
L2: take win, else block opponent's immediate win, else random
L3: take win, else block, else random among moves that don't hand the opponent
    an immediate win (fall back to all if none safe)

Uses the repo's own Connect4 njit kernels for move gen / win detection, and
cross-validates every declared result of the first 500 L0 games against an
independent pure-python 4-in-a-row scanner.

usage: baseline_ladder.py <level 0-3> <n_games> <out_csv> <seed>
"""
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero")
sys.path.insert(0, str(ROOT))

import numpy as np
from Connect4.game import Connect4


def independent_win_scan(board, player):
    """Full-board scan, written independently of the repo's check_win logic."""
    B = board
    for y in range(6):
        for x in range(7):
            if B[y][x] != player:
                continue
            for dy, dx in ((0, 1), (1, 0), (1, 1), (-1, 1)):
                n = 1
                yy, xx = y + dy, x + dx
                while 0 <= yy < 6 and 0 <= xx < 7 and B[yy][xx] == player:
                    n += 1
                    if n == 4:
                        return True
                    yy += dy
                    xx += dx
    return False


def winning_cols(game, player):
    """Columns where `player` wins immediately if they move there now."""
    out = []
    for c in game.get_legal_actions():
        b = game.board.copy()
        Connect4.do_action_MCTS(b, c, player)
        if Connect4.check_win_MCTS(b, player, np.array([c], dtype=np.int8)) == player:
            out.append(int(c))
    return out


def pick(rng, level, game):
    p = game.next_player
    legal = [int(c) for c in game.get_legal_actions()]
    if level >= 1:
        wins = winning_cols(game, p)
        if wins:
            return wins[rng.integers(len(wins))]
    if level >= 2:
        threats = winning_cols(game, -p)
        if threats:
            return threats[rng.integers(len(threats))]
    if level >= 3:
        safe = []
        for c in legal:
            b = game.board.copy()
            Connect4.do_action_MCTS(b, c, p)
            # does the opponent now have an immediate win?
            opp_win = False
            for c2 in range(7):
                if np.sum(np.abs(b[:, c2])) >= 6:
                    continue
                b2 = b.copy()
                Connect4.do_action_MCTS(b2, c2, -p)
                if Connect4.check_win_MCTS(b2, -p, np.array([c2], dtype=np.int8)) == -p:
                    opp_win = True
                    break
            if not opp_win:
                safe.append(c)
        if safe:
            return safe[rng.integers(len(safe))]
    return legal[rng.integers(len(legal))]


def main():
    level = int(sys.argv[1])
    n_games = int(sys.argv[2])
    out_csv = sys.argv[3]
    seed = int(sys.argv[4])
    rng = np.random.default_rng(seed)

    xval_games = 500 if level == 0 else 0
    xval_fail = 0

    rows = []
    for g in range(n_games):
        game = Connect4()
        winner = -2
        while winner == -2 and len(game.action_history) < 42:
            a = pick(rng, level, game)
            mover = game.next_player
            game.do_action(a)
            winner = game.check_win()
            if g < xval_games:
                indep = independent_win_scan(game.board, mover)
                declared = (winner == mover)
                if indep != declared:
                    xval_fail += 1
        if winner == -2:
            winner = 0
        rows.append((int(winner), len(game.action_history)))

    with open(out_csv, "w") as f:
        f.write("winner,length\n")
        for w, l in rows:
            f.write(f"{w},{l}\n")

    n = len(rows)
    w1 = sum(1 for w, _ in rows if w == -1)
    w2 = sum(1 for w, _ in rows if w == 1)
    dr = n - w1 - w2
    avg = sum(l for _, l in rows) / n
    print(f"L{level} n={n} p-1={w1 / n:.4f} draw={dr / n:.4f} p+1={w2 / n:.4f} avg_len={avg:.2f} "
          f"xval_fail={xval_fail}/{xval_games}")


if __name__ == "__main__":
    main()
