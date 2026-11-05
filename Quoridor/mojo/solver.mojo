# Exact alpha-beta solver for Quoridor, mirroring Quoridor/solver.py and Quoridor/cpp/solver.cpp.
# Build: mojo build -O3 -D ASSERT=none solver.mojo -o solver_bin
# Usage: solver <positions.bin> <zobrist.bin> <results.txt> <max_depth> <budget> <tt_bits>
# positions.bin: int32 count, then per position int8 player and 243 int8 board cells (3 x 9 x 9).
# zobrist.bin: 313 little-endian uint64 keys. results.txt: "proven score move nodes depth" per line, then
# "time_ns <solve time>".
from std.collections import Array
from std.sys import argv
from std.time import perf_counter_ns

comptime N = 9
comptime W = 8
comptime CELLS = 81
comptime BOARD = 243
comptime MATE = 10000
comptime INF = 30000
comptime EXACT = 1
comptime LOWER = 2
comptime UPPER = 3
comptime MAXM = 136

comptime Cell = Int
comptime Step = Int
comptime Board = Array[Cell, BOARD]
comptime Grid = Array[Int, CELLS]
comptime Dist = Array[Step, CELLS]
comptime Flags = Array[Bool, CELLS]
comptime Moves = Array[Int, MAXM]


@always_inline
def h(b: Board, r: Int, c: Int) -> Int:
    return Int(b[r * N + c])


@always_inline
def v(b: Board, r: Int, c: Int) -> Int:
    return Int(b[CELLS + r * N + c])


@always_inline
def can_move(b: Board, r: Int, c: Int, d: Int) -> Bool:
    if d == 0:
        return r > 0 and h(b, r - 1, c) == 0 and not (c > 0 and h(b, r - 1, c - 1) != 0)
    if d == 1:
        return r < N - 1 and h(b, r, c) == 0 and not (c > 0 and h(b, r, c - 1) != 0)
    if d == 2:
        return c < N - 1 and v(b, r, c) == 0 and not (r > 0 and v(b, r - 1, c) != 0)
    return c > 0 and v(b, r, c - 1) == 0 and not (r > 0 and v(b, r - 1, c - 1) != 0)


@always_inline
def step_r(r: Int, d: Int) -> Int:
    if d == 0:
        return r - 1
    if d == 1:
        return r + 1
    return r


@always_inline
def step_c(c: Int, d: Int) -> Int:
    if d == 2:
        return c + 1
    if d == 3:
        return c - 1
    return c


def dist_map(b: Board, goal_row: Int, mut dist: Dist):
    for i in range(CELLS):
        dist[i] = -1
    var qr = Grid(uninitialized=True)
    var qc = Grid(uninitialized=True)
    var head = 0
    var tail = 0
    for c in range(N):
        dist[goal_row * N + c] = 0
        qr[tail] = goal_row
        qc[tail] = c
        tail += 1
    while head < tail:
        var r = qr[head]
        var c = qc[head]
        head += 1
        for d in range(4):
            if not can_move(b, r, c, d):
                continue
            var nr = step_r(r, d)
            var nc = step_c(c, d)
            if dist[nr * N + nc] < 0:
                dist[nr * N + nc] = dist[r * N + c] + 1
                qr[tail] = nr
                qc[tail] = nc
                tail += 1


def mark_path(b: Board, dist: Dist, r0: Int, c0: Int, mut vert: Flags, mut horiz: Flags):
    var r = r0
    var c = c0
    while dist[r * N + c] > 0:
        for d in range(4):
            if not can_move(b, r, c, d):
                continue
            var nr = step_r(r, d)
            var nc = step_c(c, d)
            if dist[nr * N + nc] == dist[r * N + c] - 1:
                if d == 0:
                    vert[nr * N + nc] = True
                elif d == 1:
                    vert[r * N + c] = True
                elif d == 2:
                    horiz[r * N + c] = True
                else:
                    horiz[nr * N + nc] = True
                r = nr
                c = nc
                break


@always_inline
def goal_row(player: Int) -> Int:
    return 0 if player == -1 else N - 1


@always_inline
def pawn_r(b: Board, player: Int) -> Int:
    return Int(b[2 * CELLS + (1 if player == -1 else 2) * N])


@always_inline
def pawn_c(b: Board, player: Int) -> Int:
    return Int(b[2 * CELLS + (1 if player == -1 else 2) * N + 1])


@always_inline
def walls_left(b: Board, player: Int) -> Int:
    return Int(b[2 * CELLS + (0 if player == -1 else 1)])


def wall_fits(b: Board, t: Int, gr: Int, gc: Int) -> Bool:
    if t == 1:
        if h(b, gr, gc) != 0 or v(b, gr, gc) != 0:
            return False
        if gc > 0 and h(b, gr, gc - 1) != 0:
            return False
        if gc < N - 2 and h(b, gr, gc + 1) != 0:
            return False
        return True
    if v(b, gr, gc) != 0 or h(b, gr, gc) != 0:
        return False
    if gr > 0 and v(b, gr - 1, gc) != 0:
        return False
    if gr < N - 2 and v(b, gr + 1, gc) != 0:
        return False
    return True


@always_inline
def code(t: Int, r: Int, c: Int) -> Int:
    return (t * N + r) * N + c


def legal_actions(mut b: Board, player: Int, mut out: Moves) -> Int:
    var k = 0
    var r = pawn_r(b, player)
    var c = pawn_c(b, player)
    var orow = pawn_r(b, -player)
    var ocol = pawn_c(b, -player)
    for d in range(4):
        if not can_move(b, r, c, d):
            continue
        var nr = step_r(r, d)
        var nc = step_c(c, d)
        if nr != orow or nc != ocol:
            out[k] = code(0, nr, nc)
            k += 1
        elif can_move(b, nr, nc, d):
            out[k] = code(0, step_r(nr, d), step_c(nc, d))
            k += 1
        else:
            for d2 in range(4):
                if (d2 < 2) == (d < 2):
                    continue
                if can_move(b, nr, nc, d2):
                    out[k] = code(0, step_r(nr, d2), step_c(nc, d2))
                    k += 1
    if walls_left(b, player) > 0:
        var vert = Flags(fill=False)
        var horiz = Flags(fill=False)
        var dist = Dist(uninitialized=True)
        for s in range(2):
            var p = 2 * s - 1
            dist_map(b, goal_row(p), dist)
            mark_path(b, dist, pawn_r(b, p), pawn_c(b, p), vert, horiz)
        var br = pawn_r(b, -1)
        var bc = pawn_c(b, -1)
        var wr = pawn_r(b, 1)
        var wc = pawn_c(b, 1)
        var black = Dist(uninitialized=True)
        var white = Dist(uninitialized=True)
        for t in range(1, 3):
            var plane = (t - 1) * CELLS
            for gr in range(N - 1):
                for gc in range(N - 1):
                    if not wall_fits(b, t, gr, gc):
                        continue
                    var cuts: Bool
                    if t == 1:
                        cuts = vert[gr * N + gc] or vert[gr * N + gc + 1]
                    else:
                        cuts = horiz[gr * N + gc] or horiz[(gr + 1) * N + gc]
                    if cuts:
                        b[plane + gr * N + gc] = 1
                        dist_map(b, 0, black)
                        var ok = black[br * N + bc] >= 0
                        if ok:
                            dist_map(b, N - 1, white)
                            ok = white[wr * N + wc] >= 0
                        b[plane + gr * N + gc] = 0
                        if not ok:
                            continue
                    out[k] = code(t, gr, gc)
                    k += 1
    return k


def insert_by_key(mut moves: Moves, mut keys: Moves, mut count: Int, move: Int, key: Int):
    var i = count
    while i > 0 and keys[i - 1] > key:
        moves[i] = moves[i - 1]
        keys[i] = keys[i - 1]
        i -= 1
    moves[i] = move
    keys[i] = key
    count += 1


def ordered_moves(mut b: Board, player: Int, hint: Int, mut out: Moves) -> Int:
    var legal = Moves(uninitialized=True)
    var k = legal_actions(b, player, legal)
    var own = Dist(uninitialized=True)
    dist_map(b, goal_row(player), own)
    var current = Int(own[pawn_r(b, player) * N + pawn_c(b, player)])
    var vert = Flags(fill=False)
    var horiz = Flags(fill=False)
    if walls_left(b, player) > 0:
        var theirs = Dist(uninitialized=True)
        dist_map(b, goal_row(-player), theirs)
        mark_path(b, theirs, pawn_r(b, -player), pawn_c(b, -player), vert, horiz)
    var better = Moves(uninitialized=True)
    var better_keys = Moves(uninitialized=True)
    var nb = 0
    var worse = Moves(uninitialized=True)
    var worse_keys = Moves(uninitialized=True)
    var nw = 0
    var cutting = Moves(uninitialized=True)
    var nc = 0
    var other = Moves(uninitialized=True)
    var no = 0
    var n = 0
    for i in range(k):
        var m = legal[i]
        if m == hint:
            out[n] = m
            n += 1
            continue
        var t = m // CELLS
        var r = (m // N) % N
        var c = m % N
        if t == 0:
            var after = Int(own[r * N + c])
            if after < current:
                insert_by_key(better, better_keys, nb, m, after)
            else:
                insert_by_key(worse, worse_keys, nw, m, 200 + after)
        else:
            var cut: Bool
            if t == 1:
                cut = vert[r * N + c] or vert[r * N + c + 1]
            else:
                cut = horiz[r * N + c] or horiz[(r + 1) * N + c]
            if cut:
                cutting[nc] = m
                nc += 1
            else:
                other[no] = m
                no += 1
    for i in range(nb):
        out[n] = better[i]
        n += 1
    for i in range(nc):
        out[n] = cutting[i]
        n += 1
    for i in range(nw):
        out[n] = worse[i]
        n += 1
    for i in range(no):
        out[n] = other[i]
        n += 1
    return n


struct Table(Movable):
    var key: List[UInt64]
    var score: List[Int32]
    var depth: List[Int16]
    var flag: List[Int8]
    var move: List[Int32]
    var zob: List[UInt64]
    var mask: UInt64
    var nodes: Int
    var budget: Int
    var aborted: Bool

    def __init__(out self, size: Int, var zob: List[UInt64], budget: Int):
        self.key = List[UInt64](length=size, fill=0)
        self.score = List[Int32](length=size, fill=0)
        self.depth = List[Int16](length=size, fill=-1)
        self.flag = List[Int8](length=size, fill=0)
        self.move = List[Int32](length=size, fill=-1)
        self.zob = zob^
        self.mask = UInt64(size - 1)
        self.nodes = 0
        self.budget = budget
        self.aborted = False

    def clear(mut self):
        for i in range(len(self.key)):
            self.key[i] = 0
            self.depth[i] = -1
        self.nodes = 0
        self.aborted = False


def hash(b: Board, player: Int, zob: List[UInt64]) -> UInt64:
    var key: UInt64 = 0
    for gr in range(W):
        for gc in range(W):
            if h(b, gr, gc) != 0:
                key ^= zob[gr * W + gc]
            if v(b, gr, gc) != 0:
                key ^= zob[W * W + gr * W + gc]
    var base = 2 * W * W
    key ^= zob[base + pawn_r(b, -1) * N + pawn_c(b, -1)]
    key ^= zob[base + CELLS + pawn_r(b, 1) * N + pawn_c(b, 1)]
    base += 2 * CELLS
    key ^= zob[base + walls_left(b, -1)]
    key ^= zob[base + 11 + walls_left(b, 1)]
    if player == 1:
        key ^= zob[base + 22]
    return key


@always_inline
def to_tt(score: Int, ply: Int) -> Int:
    if score > MATE - 1000:
        return score + ply
    if score < -(MATE - 1000):
        return score - ply
    return score


@always_inline
def from_tt(score: Int, ply: Int) -> Int:
    if score > MATE - 1000:
        return score - ply
    if score < -(MATE - 1000):
        return score + ply
    return score


def search(mut b: Board, player: Int, depth: Int, alpha_in: Int, beta: Int, ply: Int, mut tt: Table) -> Int:
    tt.nodes += 1
    if tt.nodes > tt.budget:
        tt.aborted = True
        return 0
    if pawn_r(b, -player) == goal_row(-player):
        return -(MATE - ply)
    if depth == 0:
        return 0
    var alpha = alpha_in
    var key = hash(b, player, tt.zob)
    var slot = Int(key & tt.mask)
    var hint = -1
    if tt.key[slot] == key:
        hint = Int(tt.move[slot])
        if Int(tt.depth[slot]) >= depth:
            var stored = from_tt(Int(tt.score[slot]), ply)
            var flag = Int(tt.flag[slot])
            if flag == EXACT or (flag == LOWER and stored >= beta) or (flag == UPPER and stored <= alpha):
                return stored
    var moves = Moves(uninitialized=True)
    var k = ordered_moves(b, player, hint, moves)
    var original_alpha = alpha
    var best = -INF
    var best_move = -1
    var meta = 2 * CELLS + (1 if player == -1 else 2) * N
    var wall_slot = 2 * CELLS + (0 if player == -1 else 1)
    for i in range(k):
        var m = moves[i]
        var t = m // CELLS
        var r = (m // N) % N
        var c = m % N
        var old_r = 0
        var old_c = 0
        if t == 0:
            old_r = Int(b[meta])
            old_c = Int(b[meta + 1])
            b[meta] = Cell(r)
            b[meta + 1] = Cell(c)
        else:
            b[(t - 1) * CELLS + r * N + c] = 1
            b[wall_slot] -= 1
        var score = -search(b, -player, depth - 1, -beta, -alpha, ply + 1, tt)
        if t == 0:
            b[meta] = Cell(old_r)
            b[meta + 1] = Cell(old_c)
        else:
            b[(t - 1) * CELLS + r * N + c] = 0
            b[wall_slot] += 1
        if tt.aborted:
            return 0
        if score > best:
            best = score
            best_move = m
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    tt.key[slot] = key
    tt.score[slot] = Int32(to_tt(best, ply))
    tt.depth[slot] = Int16(depth)
    tt.flag[slot] = Int8(UPPER if best <= original_alpha else (LOWER if best >= beta else EXACT))
    tt.move[slot] = Int32(best_move)
    return best


def read_u64(data: List[UInt8], offset: Int) -> UInt64:
    var value: UInt64 = 0
    for i in range(8):
        value |= UInt64(data[offset + i]) << UInt64(8 * i)
    return value


def as_int8(byte: UInt8) -> Int:
    var value = Int(byte)
    return value - 256 if value > 127 else value


def main() raises:
    var args = argv()
    var positions = open(String(args[1]), "r").read_bytes()
    var zob_bytes = open(String(args[2]), "r").read_bytes()
    var max_depth = atol(String(args[4]))
    var budget = atol(String(args[5]))
    var tt_bits = atol(String(args[6]))
    var zob = List[UInt64]()
    for i in range(len(zob_bytes) // 8):
        zob.append(read_u64(zob_bytes, 8 * i))
    var tt = Table(1 << tt_bits, zob^, budget)
    var count = Int(read_u64(positions, 0) & 0xFFFFFFFF)
    var lines = String()
    var total_ns = 0
    for p in range(count):
        var offset = 4 + p * (BOARD + 1)
        var player = as_int8(positions[offset])
        var b = Board(fill=0)
        for i in range(BOARD):
            b[i] = Cell(as_int8(positions[offset + 1 + i]))
        var start = perf_counter_ns()
        tt.clear()
        var proven = 0
        var score = 0
        var move = -1
        var reached = max_depth
        for depth in range(1, max_depth + 1):
            score = search(b, player, depth, -INF, INF, 0, tt)
            if tt.aborted:
                score = 0
                reached = depth
                break
            if abs(score) > MATE - 1000:
                proven = 1
                move = Int(tt.move[Int(hash(b, player, tt.zob) & tt.mask)])
                reached = depth
                break
        total_ns += Int(perf_counter_ns() - start)
        lines += String(proven, " ", score, " ", move, " ", tt.nodes, " ", reached, "\n")
    lines += String("time_ns ", total_ns, "\n")
    with open(String(args[3]), "w") as f:
        f.write(lines)
