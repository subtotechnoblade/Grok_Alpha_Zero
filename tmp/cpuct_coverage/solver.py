"""Exact Connect4 bitboard solver (weak / sign-only), numba njit.

Standard Pons-style negamax + alpha-beta + transposition table. Only the SIGN of the
score is needed to classify a move (win / draw / loss), so every search runs with the
null window (-1, 1), which is far cheaper than a strong solve.

Bit layout: bit index = col * 7 + row, row 0 = bottom. Bit 6 of each column is the
sentinel used by the standard play()/alignment() trick.

Public entry points:
  board_to_bits(board, player) -> (position, mask, moves)
  solve_sign(position, mask, moves, node_cap) -> (sign, ok)
  classify(board, player, node_cap) -> (per-move sign array of len 7, legal mask, ok)
     per-move sign is from the MOVER's point of view: +1 win, 0 draw, -1 loss.
"""
import numpy as np
from numba import njit

W, H = 7, 6
TT_BITS = 23
TT_SIZE = 1 << TT_BITS
ORDER = np.array([3, 2, 4, 1, 5, 0, 6], dtype=np.int64)

BOTTOM = np.array([1 << (c * (H + 1)) for c in range(W)], dtype=np.uint64)
TOP = np.array([1 << (H - 1 + c * (H + 1)) for c in range(W)], dtype=np.uint64)
COLMASK = np.array([(((1 << H) - 1) << (c * (H + 1))) for c in range(W)], dtype=np.uint64)


@njit(cache=True, inline="always")
def _alignment(pos):
    # horizontal (shift by H+1 = 7)
    m = pos & (pos >> np.uint64(7))
    if m & (m >> np.uint64(14)):
        return True
    # vertical (shift by 1)
    m = pos & (pos >> np.uint64(1))
    if m & (m >> np.uint64(2)):
        return True
    # diagonal /  (shift by 6)
    m = pos & (pos >> np.uint64(6))
    if m & (m >> np.uint64(12)):
        return True
    # diagonal \  (shift by 8)
    m = pos & (pos >> np.uint64(8))
    if m & (m >> np.uint64(16)):
        return True
    return False


@njit(cache=True, inline="always")
def _can_play(mask, col, top):
    return (mask & top[col]) == np.uint64(0)


@njit(cache=True, inline="always")
def _is_winning_move(position, mask, col, bottom, colmask):
    pos = position | ((mask + bottom[col]) & colmask[col])
    return _alignment(pos)


@njit(cache=True)
def _negamax(position, mask, alpha, beta, moves, tt_key, tt_val, node_cap,
             bottom, top, colmask, order, counter):
    counter[0] += 1
    if counter[0] > node_cap:
        return 0

    if moves == W * H:
        return 0

    for i in range(W):
        c = order[i]
        if _can_play(mask, c, top) and _is_winning_move(position, mask, c, bottom, colmask):
            return (W * H + 1 - moves) // 2

    maxs = (W * H - 1 - moves) // 2
    key = position + mask
    idx = np.uint64(key % np.uint64(TT_SIZE))
    if tt_key[idx] == key + np.uint64(1):
        v = tt_val[idx]       # stored value is an UPPER bound on the true score
        if v < maxs:
            maxs = v
    if beta > maxs:
        beta = maxs
        if alpha >= beta:
            return beta

    for i in range(W):
        c = order[i]
        if _can_play(mask, c, top):
            pos2 = position ^ mask
            mask2 = mask | (mask + bottom[c])
            score = -_negamax(pos2, mask2, -beta, -alpha, moves + 1, tt_key, tt_val,
                              node_cap, bottom, top, colmask, order, counter)
            if counter[0] > node_cap:
                return 0
            if score >= beta:
                return score
            if score > alpha:
                alpha = score

    tt_key[idx] = key + np.uint64(1)
    tt_val[idx] = alpha
    return alpha


@njit(cache=True)
def _brute(position, mask, moves, bottom, top, colmask):
    """Plain exhaustive negamax, no pruning, no TT. Reference for validation only."""
    if moves == W * H:
        return 0
    best = -100
    any_move = False
    for c in range(W):
        if _can_play(mask, c, top):
            any_move = True
            if _is_winning_move(position, mask, c, bottom, colmask):
                return 1
            pos2 = position ^ mask
            mask2 = mask | (mask + bottom[c])
            s = -_brute(pos2, mask2, moves + 1, bottom, top, colmask)
            if s > best:
                best = s
            if best == 1:
                return 1
    if not any_move:
        return 0
    return best


class Solver:
    def __init__(self):
        self.tt_key = np.zeros(TT_SIZE, dtype=np.uint64)
        self.tt_val = np.zeros(TT_SIZE, dtype=np.int8)
        self.counter = np.zeros(1, dtype=np.int64)

    def reset(self):
        self.tt_key[:] = 0

    def sign(self, position, mask, moves, node_cap=3_000_000):
        """-1/0/+1 for the player to move, plus ok flag."""
        self.counter[0] = 0
        v = _negamax(np.uint64(position), np.uint64(mask), -1, 1, moves,
                     self.tt_key, self.tt_val, node_cap, BOTTOM, TOP, COLMASK,
                     ORDER, self.counter)
        ok = self.counter[0] <= node_cap
        return (int(np.sign(v)) if ok else 0), ok, int(self.counter[0])

    def brute_sign(self, position, mask, moves):
        return int(_brute(np.uint64(position), np.uint64(mask), moves, BOTTOM, TOP, COLMASK))

    def classify(self, board, player, node_cap=3_000_000):
        """For the position (board, player to move) return
             signs[7]  : +1/0/-1 from the MOVER's view for each column (999 if illegal)
             ok        : all sub-solves finished within node_cap
        """
        position, mask, moves = board_to_bits(board, player)
        signs = np.full(7, 999, dtype=np.int64)
        ok_all = True
        for c in range(W):
            if not _can_play(np.uint64(mask), c, TOP):
                continue
            if _is_winning_move(np.uint64(position), np.uint64(mask), c, BOTTOM, COLMASK):
                signs[c] = 1
                continue
            pos2 = np.uint64(position) ^ np.uint64(mask)
            mask2 = np.uint64(mask) | (np.uint64(mask) + BOTTOM[c])
            if moves + 1 == W * H:
                signs[c] = 0
                continue
            s, ok, _ = self.sign(pos2, mask2, moves + 1, node_cap)
            ok_all = ok_all and ok
            signs[c] = -s          # negate: s is from the opponent's view
        return signs, ok_all


def board_to_bits(board, player):
    """board: 6x7 int8, board[y][x], y=0 top. player: side to move (-1 or 1)."""
    position = 0
    mask = 0
    moves = 0
    for x in range(W):
        for y in range(H):
            v = board[y][x]
            if v != 0:
                r = (H - 1) - y
                bit = 1 << (x * (H + 1) + r)
                mask |= bit
                moves += 1
                if v == player:
                    position |= bit
    return position, mask, moves
