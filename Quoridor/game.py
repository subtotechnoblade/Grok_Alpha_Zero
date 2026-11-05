import numpy as np
from numba import njit

SIZE = 9
WALLS = 10
NUM_PLANES = 20
WALL_PLANES = (2, 3, 14, 15, 16, 17, 18, 19)
MAX_ACTIONS = 8 + 2 * (SIZE - 1) * (SIZE - 1)
NUM_AUX_HEADS = 5
MOVES_LEFT_SCALE = 100.0


@njit(cache=True)
def _can_move(H, V, r, c, d, n):
    """d: 0 north (row - 1), 1 south, 2 east (col + 1), 3 west."""
    if d == 0:
        return r > 0 and H[r - 1, c] == 0 and not (c > 0 and H[r - 1, c - 1] != 0)
    if d == 1:
        return r < n - 1 and H[r, c] == 0 and not (c > 0 and H[r, c - 1] != 0)
    if d == 2:
        return c < n - 1 and V[r, c] == 0 and not (r > 0 and V[r - 1, c] != 0)
    return c > 0 and V[r, c - 1] == 0 and not (r > 0 and V[r - 1, c - 1] != 0)


@njit(cache=True)
def _step(r, c, d):
    if d == 0:
        return r - 1, c
    if d == 1:
        return r + 1, c
    if d == 2:
        return r, c + 1
    return r, c - 1


@njit(cache=True)
def _dist_map(H, V, goal_row, n):
    dist = np.full((n, n), -1, dtype=np.int16)
    qr = np.empty(n * n, dtype=np.int16)
    qc = np.empty(n * n, dtype=np.int16)
    tail = 0
    for c in range(n):
        dist[goal_row, c] = 0
        qr[tail] = goal_row
        qc[tail] = c
        tail += 1
    head = 0
    while head < tail:
        r, c = qr[head], qc[head]
        head += 1
        for d in range(4):
            if _can_move(H, V, r, c, d, n):
                nr, nc = _step(r, c, d)
                if dist[nr, nc] < 0:
                    dist[nr, nc] = dist[r, c] + 1
                    qr[tail] = nr
                    qc[tail] = nc
                    tail += 1
    return dist


@njit(cache=True)
def _mark_path(H, V, dist, r, c, n, vert_used, horiz_used):
    """Walks one shortest path down dist, marking edges: vert_used[r, c] = (r, c)-(r+1, c), horiz_used[r, c] = (r, c)-(r, c+1)."""
    while dist[r, c] > 0:
        for d in range(4):
            if _can_move(H, V, r, c, d, n):
                nr, nc = _step(r, c, d)
                if dist[nr, nc] == dist[r, c] - 1:
                    if d == 0:
                        vert_used[nr, nc] = True
                    elif d == 1:
                        vert_used[r, c] = True
                    elif d == 2:
                        horiz_used[r, c] = True
                    else:
                        horiz_used[nr, nc] = True
                    r, c = nr, nc
                    break


@njit(cache=True)
def _goal_row(player, n):
    return 0 if player == -1 else n - 1


@njit(cache=True)
def _pos(board, player):
    row = 1 if player == -1 else 2
    return board[2, row, 0], board[2, row, 1]


@njit(cache=True)
def _walls_left(board, player):
    return board[2, 0, 0 if player == -1 else 1]


@njit(cache=True)
def _wall_fits(H, V, t, gr, gc, n):
    if t == 1:
        if H[gr, gc] or V[gr, gc]:
            return False
        if gc > 0 and H[gr, gc - 1]:
            return False
        if gc < n - 2 and H[gr, gc + 1]:
            return False
        return True
    if V[gr, gc] or H[gr, gc]:
        return False
    if gr > 0 and V[gr - 1, gc]:
        return False
    if gr < n - 2 and V[gr + 1, gc]:
        return False
    return True


@njit(cache=True)
def _legal_actions(board, next_player):
    n = board.shape[1]
    H, V = board[0], board[1]
    out = np.empty((8 + 2 * (n - 1) * (n - 1), 3), dtype=np.uint8)
    k = 0
    r, c = _pos(board, next_player)
    orow, ocol = _pos(board, -next_player)
    for d in range(4):
        if not _can_move(H, V, r, c, d, n):
            continue
        nr, nc = _step(r, c, d)
        if nr != orow or nc != ocol:
            out[k, 0], out[k, 1], out[k, 2] = 0, nr, nc
            k += 1
        elif _can_move(H, V, nr, nc, d, n):
            jr, jc = _step(nr, nc, d)
            out[k, 0], out[k, 1], out[k, 2] = 0, jr, jc
            k += 1
        else:
            for d2 in range(4):
                if (d2 < 2) == (d < 2):
                    continue
                if _can_move(H, V, nr, nc, d2, n):
                    jr, jc = _step(nr, nc, d2)
                    out[k, 0], out[k, 1], out[k, 2] = 0, jr, jc
                    k += 1
    if _walls_left(board, next_player) > 0:
        vert_used = np.zeros((n, n), dtype=np.bool_)
        horiz_used = np.zeros((n, n), dtype=np.bool_)
        for p in (-1, 1):
            pr, pc = _pos(board, p)
            _mark_path(H, V, _dist_map(H, V, _goal_row(p, n), n), pr, pc, n, vert_used, horiz_used)
        br, bc = _pos(board, -1)
        wr, wc = _pos(board, 1)
        for t in (1, 2):
            plane = board[t - 1]
            for gr in range(n - 1):
                for gc in range(n - 1):
                    if not _wall_fits(H, V, t, gr, gc, n):
                        continue
                    if t == 1:
                        cuts = vert_used[gr, gc] or vert_used[gr, gc + 1]
                    else:
                        cuts = horiz_used[gr, gc] or horiz_used[gr + 1, gc]
                    if cuts:
                        plane[gr, gc] = 1
                        ok = (_dist_map(H, V, 0, n)[br, bc] >= 0 and _dist_map(H, V, n - 1, n)[wr, wc] >= 0)
                        plane[gr, gc] = 0
                        if not ok:
                            continue
                    out[k, 0], out[k, 1], out[k, 2] = t, gr, gc
                    k += 1
    return out[:k]


@njit(cache=True)
def _policy_index(t, r, c, n, flip):
    if flip:
        r = n - 1 - r if t == 0 else n - 2 - r
    return (r * n + c) * 3 + t


@njit(cache=True)
def _input_state(board, to_move):
    n = board.shape[1]
    flip = to_move == 1
    H = np.zeros((n, n), dtype=np.int8)
    V = np.zeros((n, n), dtype=np.int8)
    for gr in range(n - 1):
        sr = n - 2 - gr if flip else gr
        for gc in range(n - 1):
            H[gr, gc] = board[0, sr, gc]
            V[gr, gc] = board[1, sr, gc]
    state = np.zeros((n, n, NUM_PLANES), dtype=np.int8)
    mr, mc = _pos(board, to_move)
    orow, oc = _pos(board, -to_move)
    if flip:
        mr, orow = n - 1 - mr, n - 1 - orow
    state[mr, mc, 0] = 1
    state[orow, oc, 1] = 1
    state[:, :, 2] = H
    state[:, :, 3] = V
    for r in range(n):
        for c in range(n):
            for d in range(4):
                state[r, c, 4 + d] = 0 if _can_move(H, V, r, c, d, n) else 1
    state[:, :, 8] = _walls_left(board, to_move)
    state[:, :, 9] = _walls_left(board, -to_move)
    for plane, goal_row in ((10, 0), (11, n - 1)):
        dist = _dist_map(H, V, goal_row, n)
        state[:, :, plane] = np.where(dist < 0, 127, np.minimum(dist, 127)).astype(np.int8)
    state[:, :, 12] = 1
    state[:, :, 13] = 1 if to_move == -1 else 0
    _wall_impact(H, V, mr, mc, orow, oc, n, state)
    return state


@njit(cache=True)
def _wall_impact(H, V, mr, mc, tr, tc, n, state):
    """Planes 14-19 per wall groove: H legal, V legal, my path increase (H, V), their path increase (H, V).
    Only walls cutting a current shortest path can change a distance, so only those are searched."""
    my_map = _dist_map(H, V, 0, n)
    their_map = _dist_map(H, V, n - 1, n)
    my_base, their_base = my_map[mr, mc], their_map[tr, tc]
    my_vert = np.zeros((n, n), dtype=np.bool_)
    my_horiz = np.zeros((n, n), dtype=np.bool_)
    their_vert = np.zeros((n, n), dtype=np.bool_)
    their_horiz = np.zeros((n, n), dtype=np.bool_)
    _mark_path(H, V, my_map, mr, mc, n, my_vert, my_horiz)
    _mark_path(H, V, their_map, tr, tc, n, their_vert, their_horiz)
    for t in (1, 2):
        plane = H if t == 1 else V
        for gr in range(n - 1):
            for gc in range(n - 1):
                if not _wall_fits(H, V, t, gr, gc, n):
                    continue
                if t == 1:
                    cuts_me = my_vert[gr, gc] or my_vert[gr, gc + 1]
                    cuts_them = their_vert[gr, gc] or their_vert[gr, gc + 1]
                else:
                    cuts_me = my_horiz[gr, gc] or my_horiz[gr + 1, gc]
                    cuts_them = their_horiz[gr, gc] or their_horiz[gr + 1, gc]
                my_delta = 0
                their_delta = 0
                if cuts_me or cuts_them:
                    plane[gr, gc] = 1
                    if cuts_me:
                        d = _dist_map(H, V, 0, n)[mr, mc]
                        my_delta = -1 if d < 0 else d - my_base
                    if cuts_them and my_delta >= 0:
                        d = _dist_map(H, V, n - 1, n)[tr, tc]
                        their_delta = -1 if d < 0 else d - their_base
                    plane[gr, gc] = 0
                    if my_delta < 0 or their_delta < 0:
                        continue
                state[gr, gc, 13 + t] = 1
                state[gr, gc, 15 + t] = min(my_delta, 127)
                state[gr, gc, 17 + t] = min(their_delta, 127)


def _mirror_policy_perm(n):
    perm = np.empty(3 * n * n, dtype=np.int64)
    for r in range(n):
        for c in range(n):
            for t in range(3):
                mc = n - 1 - c if t == 0 else (n - 2 - c if c < n - 1 else c)
                perm[(r * n + c) * 3 + t] = (r * n + mc) * 3 + t
    return perm


MIRROR_PERM = _mirror_policy_perm(SIZE)


def _flip_policy_perm(n):
    perm = np.empty(3 * n * n, dtype=np.int64)
    for r in range(n):
        for c in range(n):
            for t in range(3):
                fr = n - 1 - r if t == 0 else (n - 2 - r if r < n - 1 else r)
                perm[(r * n + c) * 3 + t] = (fr * n + c) * 3 + t
    return perm


FLIP_PERM = _flip_policy_perm(SIZE)


def mirror_states(states):
    """(..., n, n, NUM_PLANES) canonical states mirrored left-right."""
    out = np.flip(states, -2).copy()
    for ch in WALL_PLANES:
        out[..., :-1, ch] = states[..., -2::-1, ch]
        out[..., -1, ch] = 0
    out[..., 6], out[..., 7] = np.flip(states[..., 7], -1), np.flip(states[..., 6], -1)
    return out


class Quoridor:
    def __init__(self):
        n = SIZE
        # planes: 0 horizontal wall centers, 1 vertical wall centers (groove (r, c), r, c < n - 1),
        # 2 meta: [0] walls left (black, white), [1] black (row, col), [2] white (row, col)
        self.board = np.zeros((3, n, n), dtype=np.int8)
        self.board[2, 0, 0] = self.board[2, 0, 1] = WALLS
        self.board[2, 1, 0], self.board[2, 1, 1] = n - 1, n // 2
        self.board[2, 2, 0], self.board[2, 2, 1] = 0, n // 2
        self.next_player = -1
        self.action_history = []
        self.policy_shape = (3 * n * n,)

    def get_next_player(self):
        return self.next_player

    def pawn(self, player):
        return tuple(int(v) for v in _pos(self.board, player))

    def walls_left(self, player):
        return int(_walls_left(self.board, player))

    def input_action(self):
        while True:
            try:
                action = np.array(list(map(int, input("Action (type row col, 0 pawn 1 H wall 2 V wall):").split())), dtype=np.uint8)
                if any(np.array_equal(action, a) for a in self.get_legal_actions()):
                    return action
                print("Illegal move")
            except Exception:
                print("Invalid input")

    def get_legal_actions(self) -> np.array:
        return self.get_legal_actions_MCTS(self.board, -self.next_player, np.array(self.action_history, dtype=np.uint8))

    @staticmethod
    @njit(cache=True)
    def get_legal_actions_MCTS(board: np.array, current_player: int, action_history: np.array):
        """(n, 3) uint8 actions: (0, row, col) pawn destination, (1, row, col) / (2, row, col) horizontal / vertical wall at groove."""
        return _legal_actions(board, -current_player)

    @staticmethod
    @njit(cache=True)
    def get_legal_actions_policy_MCTS(board: np.array,
                                      current_player: int,
                                      action_history: np.array,
                                      policy: np.array,
                                      normalize=True,
                                      shuffle=False) -> (np.array, np.array):
        n = board.shape[1]
        legal_actions = _legal_actions(board, -current_player)
        flip = current_player == -1
        probs = np.empty(len(legal_actions), dtype=policy.dtype)
        for i in range(len(legal_actions)):
            probs[i] = policy[_policy_index(legal_actions[i, 0], legal_actions[i, 1], legal_actions[i, 2], n, flip)]
        if normalize:
            probs /= np.sum(probs)
        if shuffle:
            order = np.random.permutation(len(legal_actions))
            legal_actions, probs = legal_actions[order], probs[order]
        return legal_actions, probs

    def do_action(self, action) -> None:
        action = np.asarray(action, dtype=np.uint8)
        self.do_action_MCTS(self.board, action, self.next_player)
        self.next_player *= -1
        self.action_history.append(action)

    @staticmethod
    @njit(cache=True)
    def do_action_MCTS(board: np.array, action: tuple, next_player: int) -> np.array:
        t, r, c = action[0], action[1], action[2]
        if t == 0:
            row = 1 if next_player == -1 else 2
            board[2, row, 0] = r
            board[2, row, 1] = c
        else:
            board[t - 1, r, c] = 1
            board[2, 0, 0 if next_player == -1 else 1] -= 1
        return board

    def get_input_state(self) -> np.array:
        return self.get_input_state_MCTS(self.board, -self.next_player, np.array(self.action_history, dtype=np.uint8))

    @staticmethod
    @njit(cache=True)
    def get_input_state_MCTS(board: np.array, current_player: int, action_history: np.array) -> np.array:
        # canonical frame: the side to move always heads for row 0 (white's board is flipped top to bottom)
        # planes: my pawn, their pawn, H walls, V walls, blocked N/S/E/W, my walls left, their walls left,
        # distance to my goal, distance to their goal, ones, black to move
        return _input_state(board, -current_player)

    def check_win(self) -> int:
        return self.check_win_MCTS(self.board, -self.next_player, np.array(self.action_history, dtype=np.uint8))

    @staticmethod
    @njit(cache=True)
    def check_win_MCTS(board: np.array, current_player: int, action_history: np.array) -> int:
        n = board.shape[1]
        if board[2, 1, 0] == 0:
            return -1
        if board[2, 2, 0] == n - 1:
            return 1
        return -2

    num_aux_heads = NUM_AUX_HEADS
    opponent_policy_perm = FLIP_PERM

    @staticmethod
    @njit(cache=True)
    def compute_aux_targets(board: np.array, action_history: np.array, MODE=0) -> np.array:
        """(len(action_history), NUM_AUX_HEADS, 1) float32 from the FINAL board. Row i is the position
        before action i, seen by the player making action i (black moves on even i).
        Heads: moves left, my final distance to goal, their final distance, my walls left, their walls left."""
        n = board.shape[1]
        num_positions = len(action_history)
        me = np.arange(num_positions) % 2
        them = 1 - me

        # moves left
        moves_left = ((num_positions - np.arange(num_positions)) / MOVES_LEFT_SCALE).reshape(-1, 1)

        # final distance to goal, mine and theirs (colour index 0 black, 1 white)
        H, V = board[0], board[1]
        br, bc = _pos(board, -1)
        wr, wc = _pos(board, 1)
        final_dist = np.array([_dist_map(H, V, 0, n)[br, bc], _dist_map(H, V, n - 1, n)[wr, wc]], dtype=np.float64) / n
        my_dist = final_dist[me].reshape(-1, 1)
        their_dist = final_dist[them].reshape(-1, 1)

        # walls left at the end, mine and theirs
        final_walls = np.array([_walls_left(board, -1), _walls_left(board, 1)], dtype=np.float64) / WALLS
        my_walls = final_walls[me].reshape(-1, 1)
        their_walls = final_walls[them].reshape(-1, 1)

        out = np.concatenate((moves_left, my_dist, their_dist, my_walls, their_walls), axis=1)
        return out.astype(np.float32).reshape(num_positions, NUM_AUX_HEADS, 1)

    def compute_policy_improvement(self, statistics):
        n = self.board.shape[1]
        flip = self.next_player == 1
        new_policy = np.zeros(self.policy_shape, dtype=np.float32)
        for (t, r, c), prob in statistics:
            new_policy[_policy_index(int(t), int(r), int(c), n, flip)] = prob
        return new_policy

    def augment_sample(self, input_states, policies):
        states = np.stack([input_states, mirror_states(input_states)]).astype(self.board.dtype)
        policies = np.asarray(policies, dtype=np.float32)
        return states, np.stack([policies, policies[:, MIRROR_PERM]])

    num_symmetries = 2

    @staticmethod
    def apply_symmetry(states, ids):
        """(n, size, size, c) states, symmetry id per row: 0 identity, 1 left-right mirror."""
        out = states.copy()
        rows = ids == 1
        out[rows] = mirror_states(states[rows])
        return out

    @staticmethod
    def undo_symmetry(policies, ids):
        out = policies.copy()
        rows = ids == 1
        out[rows] = policies[rows][:, MIRROR_PERM]
        return out


if __name__ == "__main__":
    from Game_Tester import Game_Tester

    Game_Tester(Quoridor).test()
