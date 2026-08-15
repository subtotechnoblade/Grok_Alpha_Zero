"""Validate the bitboard solver.

1. board_to_bits + _alignment agree with Connect4.check_win_MCTS on random playouts.
2. weak solve sign == exhaustive brute-force sign on random positions (deep enough that
   brute force terminates).
3. timing of a full weak solve as a function of ply, to pick a node cap / ply floor.
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp/cpuct_coverage"))

from Connect4.game import Connect4
from solver import Solver, board_to_bits, _alignment, W, H, BOTTOM, TOP, COLMASK


def random_position(rng, target_ply, max_tries=200):
    """random legal position at exactly target_ply with nobody having won yet"""
    for _ in range(max_tries):
        g = Connect4()
        ok = True
        for _ in range(target_ply):
            ls = [x for x in range(7) if np.abs(g.board[:, x]).sum() < 6]
            if not ls:
                ok = False
                break
            g.do_action(int(rng.choice(ls)))
            if g.check_win() != -2:
                ok = False
                break
        if ok and len(g.action_history) == target_ply:
            return g
    return None


def test_alignment(rng, n=400):
    """after every move, _alignment on the mover's bitboard must match check_win"""
    bad = 0
    tot = 0
    for _ in range(n):
        g = Connect4()
        while True:
            ls = [x for x in range(7) if np.abs(g.board[:, x]).sum() < 6]
            if not ls:
                break
            mover = g.next_player
            g.do_action(int(rng.choice(ls)))
            w = g.check_win()
            # bits from the perspective of the player who just moved
            pos, mask, moves = board_to_bits(g.board, mover)
            al = bool(_alignment(np.uint64(pos)))
            tot += 1
            if al != (w == mover):
                bad += 1
            if w != -2:
                break
    print(f"[1] alignment vs check_win: {tot} checks, {bad} mismatches -> "
          f"{'PASS' if bad == 0 else 'FAIL'}")
    return bad == 0


def test_vs_brute(rng, plies=(30, 28, 26, 24), per_ply=25):
    s = Solver()
    bad = 0
    tot = 0
    for ply in plies:
        for _ in range(per_ply):
            g = random_position(rng, ply)
            if g is None:
                continue
            pos, mask, moves = board_to_bits(g.board, g.next_player)
            bs = s.brute_sign(pos, mask, moves)
            ws, ok, nodes = s.sign(pos, mask, moves, node_cap=10 ** 9)
            tot += 1
            if bs != ws:
                bad += 1
                print(f"    MISMATCH ply={ply} brute={bs} weak={ws}\n{g.board}")
    print(f"[2] weak solve vs exhaustive brute force: {tot} positions, {bad} mismatches "
          f"-> {'PASS' if bad == 0 else 'FAIL'}")
    return bad == 0


def test_known():
    """the empty 6x7 board is a first-player win; centre is the only winning first move
    (classical result). Solving from empty is expensive, so only assert the sign of the
    centre-opening position (player 2 to move must be losing)."""
    s = Solver()
    g = Connect4()
    g.do_action(3)
    pos, mask, moves = board_to_bits(g.board, g.next_player)
    t0 = time.time()
    sg, ok, nodes = s.sign(pos, mask, moves, node_cap=400_000_000)
    dt = time.time() - t0
    print(f"[3] after 1.d3, player-to-move sign = {sg} (expect -1, i.e. p1 wins) "
          f"nodes={nodes:,} {dt:.1f}s -> {'PASS' if (ok and sg == -1) else 'FAIL/CAP'}")
    return ok and sg == -1


def test_timing(rng, plies=(10, 12, 14, 16, 18, 20, 22), per_ply=8):
    s = Solver()
    print(f"[4] timing of classify() (7 sub-solves) by ply, TT persists across positions")
    print(f"{'ply':>4} {'median s':>10} {'max s':>9} {'median nodes':>14}")
    for ply in plies:
        ts, ns = [], []
        for _ in range(per_ply):
            g = random_position(rng, ply)
            if g is None:
                continue
            s.counter[0] = 0
            t0 = time.time()
            signs, ok = s.classify(g.board, g.next_player, node_cap=200_000_000)
            ts.append(time.time() - t0)
            ns.append(int(s.counter[0]))
        if ts:
            print(f"{ply:>4} {np.median(ts):>10.3f} {max(ts):>9.3f} {np.median(ns):>14,.0f}")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a = test_alignment(rng)
    b = test_vs_brute(rng)
    test_timing(rng)
    c = test_known()
    print(f"\nOVERALL: {'PASS' if (a and b and c) else 'CHECK ABOVE'}")
