"""Second, independent validation pass on the solver.

A. brute force vs weak solve at shallower plies (22, 23) - more search to get wrong.
B. propagation consistency: sign(P) must equal -sign(P after the best move), for
   positions solved independently (different sub-trees, so not tautological).
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp/cpuct_coverage"))

from Connect4.game import Connect4
from solver import Solver, board_to_bits

from validate_solver import random_position


def main():
    rng = np.random.default_rng(7)
    s = Solver()

    print("A. brute force vs weak solve, plies 20-23")
    bad = tot = 0
    t0 = time.time()
    for ply in (23, 22, 21, 20):
        for _ in range(15):
            g = random_position(rng, ply)
            if g is None:
                continue
            pos, mask, moves = board_to_bits(g.board, g.next_player)
            bs = s.brute_sign(pos, mask, moves)
            ws, ok, _ = s.sign(pos, mask, moves, node_cap=10 ** 9)
            tot += 1
            if bs != ws:
                bad += 1
                print(f"   MISMATCH ply={ply} brute={bs} weak={ws}")
    print(f"   {tot} positions, {bad} mismatches, {time.time() - t0:.0f}s -> "
          f"{'PASS' if bad == 0 else 'FAIL'}")

    print("B. propagation consistency, plies 14-20")
    bad2 = tot2 = 0
    for ply in (14, 16, 18, 20):
        for _ in range(15):
            g = random_position(rng, ply)
            if g is None:
                continue
            signs, ok = s.classify(g.board, g.next_player, node_cap=100_000_000)
            if not ok:
                continue
            legal = signs != 999
            best = signs[legal].max()
            # play a best move, solve the child independently, expect -best
            cand = [c for c in range(7) if signs[c] == best]
            c = int(rng.choice(cand))
            g2 = Connect4()
            g2.board = g.board.copy()
            g2.next_player = g.next_player
            g2.action_history = list(g.action_history)
            g2.do_action(c)
            if g2.check_win() != -2:
                child = -1  # child position is a loss for the player to move there
            else:
                pos, mask, moves = board_to_bits(g2.board, g2.next_player)
                cs, ok2, _ = s.sign(pos, mask, moves, node_cap=100_000_000)
                if not ok2:
                    continue
                child = cs
            tot2 += 1
            if child != -best:
                bad2 += 1
                print(f"   MISMATCH ply={ply} best={best} child_sign={child}")
    print(f"   {tot2} positions, {bad2} mismatches -> {'PASS' if bad2 == 0 else 'FAIL'}")


if __name__ == "__main__":
    main()
