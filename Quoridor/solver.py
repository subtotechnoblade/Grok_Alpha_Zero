"""Exact alpha-beta solver for Quoridor positions with few walls left (numba).

Negamax with iterative deepening, a Zobrist-hashed transposition table and distance-to-win scores: a win in n
plies from the root scores MATE - n. A position is solved when the root score is a proven win or loss within
max_depth plies and node_budget nodes; otherwise the result is unknown and the caller should search instead.
Quoridor/cpp/solver.cpp implements the identical algorithm (same move order, table and keys).
"""
import numpy as np
from numba import njit

from Quoridor.game import SIZE, _dist_map, _goal_row, _legal_actions, _mark_path, _pos, _walls_left, _wall_fits

MATE = 10000
INF = 30000
EXACT, LOWER, UPPER = 1, 2, 3
WALL_SLOTS = (SIZE - 1) * (SIZE - 1)
ZOBRIST_SIZE = 2 * WALL_SLOTS + 2 * SIZE * SIZE + 2 * 11 + 1


def zobrist_table(seed=20261105):
    return np.random.default_rng(seed).integers(0, 2 ** 63, ZOBRIST_SIZE, dtype=np.int64).astype(np.uint64)


@njit(cache=True)
def _hash(board, player, zob):
    n = board.shape[1]
    w = n - 1
    key = np.uint64(0)
    for gr in range(w):
        for gc in range(w):
            if board[0, gr, gc]:
                key ^= zob[gr * w + gc]
            if board[1, gr, gc]:
                key ^= zob[w * w + gr * w + gc]
    base = 2 * w * w
    br, bc = _pos(board, -1)
    wr, wc = _pos(board, 1)
    key ^= zob[base + br * n + bc]
    key ^= zob[base + n * n + wr * n + wc]
    base += 2 * n * n
    key ^= zob[base + _walls_left(board, -1)]
    key ^= zob[base + 11 + _walls_left(board, 1)]
    if player == 1:
        key ^= zob[base + 22]
    return key


@njit(cache=True)
def _move_code(t, r, c, n):
    return (t * n + r) * n + c


@njit(cache=True)
def _ordered_moves(board, player, tt_move):
    n = board.shape[1]
    H, V = board[0], board[1]
    moves = _legal_actions(board, player)
    k = len(moves)
    own_map = _dist_map(H, V, _goal_row(player, n), n)
    r, c = _pos(board, player)
    current = own_map[r, c]
    their_vert = np.zeros((n, n), dtype=np.bool_)
    their_horiz = np.zeros((n, n), dtype=np.bool_)
    if _walls_left(board, player) > 0:
        tr, tc = _pos(board, -player)
        _mark_path(H, V, _dist_map(H, V, _goal_row(-player, n), n), tr, tc, n, their_vert, their_horiz)
    keys = np.empty(k, dtype=np.int64)
    for i in range(k):
        t, mr, mc = moves[i, 0], moves[i, 1], moves[i, 2]
        if _move_code(t, mr, mc, n) == tt_move:
            keys[i] = -1
        elif t == 0:
            after = own_map[mr, mc]
            keys[i] = after if after < current else 200 + after
        elif t == 1:
            keys[i] = 100 if (their_vert[mr, mc] or their_vert[mr, mc + 1]) else 400
        else:
            keys[i] = 100 if (their_horiz[mr, mc] or their_horiz[mr + 1, mc]) else 400
    return moves[np.argsort(keys, kind="mergesort")]


@njit(cache=True)
def _to_tt(score, ply):
    if score > MATE - 1000:
        return score + ply
    if score < -(MATE - 1000):
        return score - ply
    return score


@njit(cache=True)
def _from_tt(score, ply):
    if score > MATE - 1000:
        return score - ply
    if score < -(MATE - 1000):
        return score + ply
    return score


@njit
def _search(board, player, depth, alpha, beta, ply, tt_key, tt_score, tt_depth, tt_flag, tt_move, zob, stats,
            budget):
    stats[0] += 1
    if stats[0] > budget:
        stats[1] = 1
        return 0
    n = board.shape[1]
    orow, ocol = _pos(board, -player)
    if orow == _goal_row(-player, n):
        return -(MATE - ply)
    if depth == 0:
        return 0
    key = _hash(board, player, zob)
    slot = np.int64(key & np.uint64(len(tt_key) - 1))
    best_hint = -1
    if tt_key[slot] == key:
        best_hint = tt_move[slot]
        if tt_depth[slot] >= depth:
            stored = _from_tt(tt_score[slot], ply)
            flag = tt_flag[slot]
            if flag == EXACT or (flag == LOWER and stored >= beta) or (flag == UPPER and stored <= alpha):
                return stored
    moves = _ordered_moves(board, player, best_hint)
    original_alpha = alpha
    best = -INF
    best_move = -1
    meta_row = 1 if player == -1 else 2
    wall_col = 0 if player == -1 else 1
    for i in range(len(moves)):
        t, r, c = moves[i, 0], moves[i, 1], moves[i, 2]
        if t == 0:
            old_r, old_c = board[2, meta_row, 0], board[2, meta_row, 1]
            board[2, meta_row, 0] = r
            board[2, meta_row, 1] = c
        else:
            board[t - 1, r, c] = 1
            board[2, 0, wall_col] -= 1
        score = -_search(board, -player, depth - 1, -beta, -alpha, ply + 1, tt_key, tt_score, tt_depth, tt_flag,
                         tt_move, zob, stats, budget)
        if t == 0:
            board[2, meta_row, 0] = old_r
            board[2, meta_row, 1] = old_c
        else:
            board[t - 1, r, c] = 0
            board[2, 0, wall_col] += 1
        if stats[1]:
            return 0
        if score > best:
            best = score
            best_move = _move_code(t, r, c, n)
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    tt_key[slot] = key
    tt_score[slot] = _to_tt(best, ply)
    tt_depth[slot] = depth
    tt_flag[slot] = UPPER if best <= original_alpha else (LOWER if best >= beta else EXACT)
    tt_move[slot] = best_move
    return best


@njit
def solve_kernel(board, player, max_depth, budget, tt_key, tt_score, tt_depth, tt_flag, tt_move, zob):
    """(proven, score, best move code, nodes, depth). Scores are from the player to move."""
    tt_key[:] = 0
    tt_depth[:] = -1
    stats = np.zeros(2, dtype=np.int64)
    board = board.copy()
    score = 0
    for depth in range(1, max_depth + 1):
        score = _search(board, player, depth, -INF, INF, 0, tt_key, tt_score, tt_depth, tt_flag, tt_move, zob,
                        stats, budget)
        if stats[1]:
            return False, 0, -1, stats[0], depth
        if abs(score) > MATE - 1000:
            slot = np.int64(_hash(board, player, zob) & np.uint64(len(tt_key) - 1))
            return True, score, tt_move[slot], stats[0], depth
    return False, score, -1, stats[0], max_depth


class Solver:
    def __init__(self, tt_bits=20, max_depth=60, node_budget=2_000_000):
        size = 1 << tt_bits
        self.tt_key = np.zeros(size, dtype=np.uint64)
        self.tt_score = np.zeros(size, dtype=np.int32)
        self.tt_depth = np.full(size, -1, dtype=np.int16)
        self.tt_flag = np.zeros(size, dtype=np.int8)
        self.tt_move = np.full(size, -1, dtype=np.int32)
        self.zob = zobrist_table()
        self.max_depth = max_depth
        self.node_budget = node_budget

    def raw(self, board, player):
        """(proven, score, best move code, nodes, depth)."""
        return solve_kernel(board, player, self.max_depth, self.node_budget, self.tt_key, self.tt_score,
                            self.tt_depth, self.tt_flag, self.tt_move, self.zob)

    def solve(self, board, player):
        """None if unproven, else (value +1/-1 for the player to move, plies to the end, best action, nodes)."""
        proven, score, move, nodes, depth = self.raw(board, player)
        if not proven:
            return None
        n = board.shape[1]
        action = np.array([move // (n * n), (move // n) % n, move % n], dtype=np.uint8)
        return (1 if score > 0 else -1), MATE - abs(int(score)), action, int(nodes)


_race_solver = None


def race_targets(board, player):
    """None unless neither player has walls left and the race is proven. Otherwise (value +1/-1 for the player to
    move, plies to the end, every move that keeps that result in that many plies)."""
    if _walls_left(board, -1) or _walls_left(board, 1):
        return None
    global _race_solver
    if _race_solver is None:
        _race_solver = Solver(tt_bits=16, max_depth=60, node_budget=200_000)
    root = _race_solver.solve(board, player)
    if root is None:
        return None
    value, plies, best, _ = root
    goal = _goal_row(player, board.shape[1])
    meta_row = 1 if player == -1 else 2
    keep = []
    for action in _legal_actions(board, player):
        if action[1] == goal:
            result = (1, 1)
        else:
            child = board.copy()
            child[2, meta_row, 0], child[2, meta_row, 1] = action[1], action[2]
            follow = _race_solver.solve(child, -player)
            if follow is None:
                continue
            result = (-follow[0], follow[1] + 1)
        if result == (value, plies):
            keep.append(action)
    return value, plies, keep or [best]


class CppSolver(Solver):
    """The same solver compiled from Quoridor/cpp/solver.cpp, called through ctypes with the same table and keys."""

    def __init__(self, library=None, **kwargs):
        super().__init__(**kwargs)
        import ctypes
        from pathlib import Path
        self.lib = ctypes.CDLL(str(library or Path(__file__).resolve().parent / "cpp" / "libsolver.so"))
        pointer, c_int, c_int64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64
        self.lib.solve.argtypes = [pointer, c_int, c_int, c_int64, pointer, pointer, pointer, pointer, pointer,
                                   c_int64, pointer, pointer, pointer]
        self.lib.solve.restype = c_int
        self.out = np.zeros(4, dtype=np.int32)
        self.nodes = np.zeros(1, dtype=np.int64)

    def raw(self, board, player):
        board = np.ascontiguousarray(board, dtype=np.int8)
        ptr = lambda a: a.ctypes.data
        self.lib.solve(ptr(board), int(player), self.max_depth, self.node_budget, ptr(self.tt_key),
                       ptr(self.tt_score), ptr(self.tt_depth), ptr(self.tt_flag), ptr(self.tt_move), len(self.tt_key),
                       ptr(self.zob), ptr(self.out), ptr(self.nodes))
        return bool(self.out[0]), int(self.out[1]), int(self.out[2]), int(self.nodes[0]), int(self.out[3])
