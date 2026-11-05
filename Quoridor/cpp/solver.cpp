// Exact alpha-beta solver for Quoridor, mirroring Quoridor/solver.py move for move.
// Build: g++ -O3 -march=native -shared -fPIC -o libsolver.so solver.cpp
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int N = 9;
constexpr int W = N - 1;
constexpr int MATE = 10000;
constexpr int INF = 30000;
constexpr int8_t EXACT = 1, LOWER = 2, UPPER = 3;
constexpr int MAX_MOVES = 8 + 2 * W * W;

struct Board {
    int8_t b[3][N][N];
};

struct Move {
    uint8_t t, r, c;
};

struct Table {
    uint64_t* key;
    int32_t* score;
    int16_t* depth;
    int8_t* flag;
    int32_t* move;
    uint64_t mask;
    const uint64_t* zob;
    int64_t nodes;
    int64_t budget;
    bool aborted;
};

inline bool can_move(const int8_t H[N][N], const int8_t V[N][N], int r, int c, int d) {
    if (d == 0) return r > 0 && H[r - 1][c] == 0 && !(c > 0 && H[r - 1][c - 1] != 0);
    if (d == 1) return r < N - 1 && H[r][c] == 0 && !(c > 0 && H[r][c - 1] != 0);
    if (d == 2) return c < N - 1 && V[r][c] == 0 && !(r > 0 && V[r - 1][c] != 0);
    return c > 0 && V[r][c - 1] == 0 && !(r > 0 && V[r - 1][c - 1] != 0);
}

inline void step(int r, int c, int d, int& nr, int& nc) {
    nr = r; nc = c;
    if (d == 0) nr = r - 1;
    else if (d == 1) nr = r + 1;
    else if (d == 2) nc = c + 1;
    else nc = c - 1;
}

void dist_map(const int8_t H[N][N], const int8_t V[N][N], int goal_row, int16_t dist[N][N]) {
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c) dist[r][c] = -1;
    int qr[N * N], qc[N * N], head = 0, tail = 0;
    for (int c = 0; c < N; ++c) {
        dist[goal_row][c] = 0;
        qr[tail] = goal_row; qc[tail] = c; ++tail;
    }
    while (head < tail) {
        int r = qr[head], c = qc[head]; ++head;
        for (int d = 0; d < 4; ++d) {
            if (!can_move(H, V, r, c, d)) continue;
            int nr, nc;
            step(r, c, d, nr, nc);
            if (dist[nr][nc] < 0) {
                dist[nr][nc] = dist[r][c] + 1;
                qr[tail] = nr; qc[tail] = nc; ++tail;
            }
        }
    }
}

void mark_path(const int8_t H[N][N], const int8_t V[N][N], const int16_t dist[N][N], int r, int c,
               bool vert[N][N], bool horiz[N][N]) {
    while (dist[r][c] > 0) {
        for (int d = 0; d < 4; ++d) {
            if (!can_move(H, V, r, c, d)) continue;
            int nr, nc;
            step(r, c, d, nr, nc);
            if (dist[nr][nc] == dist[r][c] - 1) {
                if (d == 0) vert[nr][nc] = true;
                else if (d == 1) vert[r][c] = true;
                else if (d == 2) horiz[r][c] = true;
                else horiz[nr][nc] = true;
                r = nr; c = nc;
                break;
            }
        }
    }
}

inline int goal_row(int player) { return player == -1 ? 0 : N - 1; }
inline int pawn_row(const Board& b, int player) { return b.b[2][player == -1 ? 1 : 2][0]; }
inline int pawn_col(const Board& b, int player) { return b.b[2][player == -1 ? 1 : 2][1]; }
inline int walls_left(const Board& b, int player) { return b.b[2][0][player == -1 ? 0 : 1]; }

bool wall_fits(const int8_t H[N][N], const int8_t V[N][N], int t, int gr, int gc) {
    if (t == 1) {
        if (H[gr][gc] || V[gr][gc]) return false;
        if (gc > 0 && H[gr][gc - 1]) return false;
        if (gc < N - 2 && H[gr][gc + 1]) return false;
        return true;
    }
    if (V[gr][gc] || H[gr][gc]) return false;
    if (gr > 0 && V[gr - 1][gc]) return false;
    if (gr < N - 2 && V[gr + 1][gc]) return false;
    return true;
}

int legal_actions(Board& b, int player, Move* out) {
    auto& H = b.b[0];
    auto& V = b.b[1];
    int k = 0;
    int r = pawn_row(b, player), c = pawn_col(b, player);
    int orow = pawn_row(b, -player), ocol = pawn_col(b, -player);
    for (int d = 0; d < 4; ++d) {
        if (!can_move(H, V, r, c, d)) continue;
        int nr, nc;
        step(r, c, d, nr, nc);
        if (nr != orow || nc != ocol) {
            out[k++] = {0, (uint8_t)nr, (uint8_t)nc};
        } else if (can_move(H, V, nr, nc, d)) {
            int jr, jc;
            step(nr, nc, d, jr, jc);
            out[k++] = {0, (uint8_t)jr, (uint8_t)jc};
        } else {
            for (int d2 = 0; d2 < 4; ++d2) {
                if ((d2 < 2) == (d < 2)) continue;
                if (can_move(H, V, nr, nc, d2)) {
                    int jr, jc;
                    step(nr, nc, d2, jr, jc);
                    out[k++] = {0, (uint8_t)jr, (uint8_t)jc};
                }
            }
        }
    }
    if (walls_left(b, player) > 0) {
        bool vert[N][N] = {}, horiz[N][N] = {};
        int16_t dist[N][N];
        for (int p : {-1, 1}) {
            dist_map(H, V, goal_row(p), dist);
            mark_path(H, V, dist, pawn_row(b, p), pawn_col(b, p), vert, horiz);
        }
        int br = pawn_row(b, -1), bc = pawn_col(b, -1), wr = pawn_row(b, 1), wc = pawn_col(b, 1);
        for (int t = 1; t <= 2; ++t) {
            auto& plane = b.b[t - 1];
            for (int gr = 0; gr < N - 1; ++gr) {
                for (int gc = 0; gc < N - 1; ++gc) {
                    if (!wall_fits(H, V, t, gr, gc)) continue;
                    bool cuts = t == 1 ? (vert[gr][gc] || vert[gr][gc + 1]) : (horiz[gr][gc] || horiz[gr + 1][gc]);
                    if (cuts) {
                        plane[gr][gc] = 1;
                        int16_t black[N][N], white[N][N];
                        dist_map(H, V, 0, black);
                        bool ok = black[br][bc] >= 0;
                        if (ok) {
                            dist_map(H, V, N - 1, white);
                            ok = white[wr][wc] >= 0;
                        }
                        plane[gr][gc] = 0;
                        if (!ok) continue;
                    }
                    out[k++] = {(uint8_t)t, (uint8_t)gr, (uint8_t)gc};
                }
            }
        }
    }
    return k;
}

inline int move_code(const Move& m) { return (m.t * N + m.r) * N + m.c; }

uint64_t hash(const Board& b, int player, const uint64_t* zob) {
    uint64_t key = 0;
    for (int gr = 0; gr < W; ++gr)
        for (int gc = 0; gc < W; ++gc) {
            if (b.b[0][gr][gc]) key ^= zob[gr * W + gc];
            if (b.b[1][gr][gc]) key ^= zob[W * W + gr * W + gc];
        }
    int base = 2 * W * W;
    key ^= zob[base + pawn_row(b, -1) * N + pawn_col(b, -1)];
    key ^= zob[base + N * N + pawn_row(b, 1) * N + pawn_col(b, 1)];
    base += 2 * N * N;
    key ^= zob[base + walls_left(b, -1)];
    key ^= zob[base + 11 + walls_left(b, 1)];
    if (player == 1) key ^= zob[base + 22];
    return key;
}

int ordered_moves(Board& b, int player, int tt_move, Move* moves) {
    Move legal[MAX_MOVES];
    int k = legal_actions(b, player, legal);
    auto& H = b.b[0];
    auto& V = b.b[1];
    int16_t own[N][N];
    dist_map(H, V, goal_row(player), own);
    int current = own[pawn_row(b, player)][pawn_col(b, player)];
    bool vert[N][N] = {}, horiz[N][N] = {};
    if (walls_left(b, player) > 0) {
        int16_t theirs[N][N];
        dist_map(H, V, goal_row(-player), theirs);
        mark_path(H, V, theirs, pawn_row(b, -player), pawn_col(b, -player), vert, horiz);
    }
    int64_t keys[MAX_MOVES];
    int order[MAX_MOVES];
    for (int i = 0; i < k; ++i) {
        const Move& m = legal[i];
        order[i] = i;
        if (move_code(m) == tt_move) keys[i] = -1;
        else if (m.t == 0) {
            int after = own[m.r][m.c];
            keys[i] = after < current ? after : 200 + after;
        } else if (m.t == 1) keys[i] = (vert[m.r][m.c] || vert[m.r][m.c + 1]) ? 100 : 400;
        else keys[i] = (horiz[m.r][m.c] || horiz[m.r + 1][m.c]) ? 100 : 400;
    }
    std::stable_sort(order, order + k, [&](int a, int c) { return keys[a] < keys[c]; });
    for (int i = 0; i < k; ++i) moves[i] = legal[order[i]];
    return k;
}

inline int to_tt(int score, int ply) {
    if (score > MATE - 1000) return score + ply;
    if (score < -(MATE - 1000)) return score - ply;
    return score;
}

inline int from_tt(int score, int ply) {
    if (score > MATE - 1000) return score - ply;
    if (score < -(MATE - 1000)) return score + ply;
    return score;
}

int search(Board& b, int player, int depth, int alpha, int beta, int ply, Table& tt) {
    if (++tt.nodes > tt.budget) {
        tt.aborted = true;
        return 0;
    }
    if (pawn_row(b, -player) == goal_row(-player)) return -(MATE - ply);
    if (depth == 0) return 0;
    uint64_t key = hash(b, player, tt.zob);
    uint64_t slot = key & tt.mask;
    int hint = -1;
    if (tt.key[slot] == key) {
        hint = tt.move[slot];
        if (tt.depth[slot] >= depth) {
            int stored = from_tt(tt.score[slot], ply);
            int8_t flag = tt.flag[slot];
            if (flag == EXACT || (flag == LOWER && stored >= beta) || (flag == UPPER && stored <= alpha)) return stored;
        }
    }
    Move moves[MAX_MOVES];
    int k = ordered_moves(b, player, hint, moves);
    int original_alpha = alpha, best = -INF, best_move = -1;
    int meta_row = player == -1 ? 1 : 2, wall_col = player == -1 ? 0 : 1;
    for (int i = 0; i < k; ++i) {
        const Move& m = moves[i];
        int8_t old_r = 0, old_c = 0;
        if (m.t == 0) {
            old_r = b.b[2][meta_row][0];
            old_c = b.b[2][meta_row][1];
            b.b[2][meta_row][0] = m.r;
            b.b[2][meta_row][1] = m.c;
        } else {
            b.b[m.t - 1][m.r][m.c] = 1;
            b.b[2][0][wall_col] -= 1;
        }
        int score = -search(b, -player, depth - 1, -beta, -alpha, ply + 1, tt);
        if (m.t == 0) {
            b.b[2][meta_row][0] = old_r;
            b.b[2][meta_row][1] = old_c;
        } else {
            b.b[m.t - 1][m.r][m.c] = 0;
            b.b[2][0][wall_col] += 1;
        }
        if (tt.aborted) return 0;
        if (score > best) {
            best = score;
            best_move = move_code(m);
        }
        if (score > alpha) alpha = score;
        if (alpha >= beta) break;
    }
    tt.key[slot] = key;
    tt.score[slot] = to_tt(best, ply);
    tt.depth[slot] = depth;
    tt.flag[slot] = best <= original_alpha ? UPPER : (best >= beta ? LOWER : EXACT);
    tt.move[slot] = best_move;
    return best;
}

}  // namespace

extern "C" int solve(const int8_t* board_in, int player, int max_depth, int64_t budget, uint64_t* tt_key,
                     int32_t* tt_score, int16_t* tt_depth, int8_t* tt_flag, int32_t* tt_move, int64_t tt_size,
                     const uint64_t* zob, int32_t* out, int64_t* nodes) {
    Board b;
    std::memcpy(b.b, board_in, sizeof(b.b));
    std::memset(tt_key, 0, sizeof(uint64_t) * tt_size);
    for (int64_t i = 0; i < tt_size; ++i) tt_depth[i] = -1;
    Table tt{tt_key, tt_score, tt_depth, tt_flag, tt_move, (uint64_t)(tt_size - 1), zob, 0, budget, false};
    int score = 0;
    for (int depth = 1; depth <= max_depth; ++depth) {
        score = search(b, player, depth, -INF, INF, 0, tt);
        if (tt.aborted) {
            out[0] = 0; out[1] = 0; out[2] = -1; out[3] = depth;
            *nodes = tt.nodes;
            return 0;
        }
        if (std::abs(score) > MATE - 1000) {
            uint64_t slot = hash(b, player, zob) & tt.mask;
            out[0] = 1; out[1] = score; out[2] = tt_move[slot]; out[3] = depth;
            *nodes = tt.nodes;
            return 1;
        }
    }
    out[0] = 0; out[1] = score; out[2] = -1; out[3] = max_depth;
    *nodes = tt.nodes;
    return 0;
}
