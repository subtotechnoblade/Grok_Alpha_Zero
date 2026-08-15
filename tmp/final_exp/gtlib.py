"""Shared ground-truth helpers for the value-head experiment.

MOVER CONVENTION (verified empirically, tmp/final_exp/check_convention.py):
Connect4.game.get_input_state stores board_state[0] = -next_player, so channel
0 is the player who JUST moved and the player TO MOVE is -chan0. The first
mover plays -1 stones. On the last stored state of a game the mover has a
win-in-1 86.3% of the time under this rule and only 60.3% under the rule in
tmp/ab_heads.py (which reads +1 as the first mover and is therefore inverted on
every EVEN ply -- half of all positions).
"""
import numpy as np
import h5py as h5

ROWS, COLS = 6, 7
PLY_BANDS = [(0, 5), (6, 9), (10, 13), (14, 17), (18, 99)]
BAND_NAMES = ["0-5", "6-9", "10-13", "14-17", "18+"]


def mover_of(state):
    """Player to move for a stored (6,7,4) state. -chan0, verified."""
    return -int(state[0, 0, 0])


def landing(board, c):
    for rr in range(ROWS - 1, -1, -1):
        if board[rr][c] == 0:
            return rr
    return -1


def win_cols(board, p):
    """Columns where dropping a p stone completes 4-in-a-row."""
    out = []
    for c in range(COLS):
        r = landing(board, c)
        if r < 0:
            continue
        b = board.copy()
        b[r][c] = p
        for dr, dc in ((0, 1), (1, 0), (1, 1), (-1, 1)):
            n = 1
            for s in (1, -1):
                rr, cc = r + dr * s, c + dc * s
                while 0 <= rr < ROWS and 0 <= cc < COLS and b[rr][cc] == p:
                    n += 1
                    rr += dr * s
                    cc += dc * s
            if n >= 4:
                out.append(c)
                break
    return out


def legal_mask(board):
    return np.array([landing(board, c) >= 0 for c in range(COLS)], dtype=bool)


def ply_of(board):
    return int((board != 0).sum())


def band_of(ply):
    for i, (lo, hi) in enumerate(PLY_BANDS):
        if lo <= ply <= hi:
            return i
    return len(PLY_BANDS) - 1


def iter_states(path, stride=1, max_games=None):
    """Yield (state, policy_target, value_target) for unaugmented game 0 of each
    stored game. Self_Play writes `aug` consecutive dataset triples per game;
    dataset_id = game_id * aug picks the unaugmented one."""
    with h5.File(path, "r") as f:
        n = int(f["game_stats"][2])
        n_sets = (len(f.keys()) - 1) // 3
        if n <= 0 or n_sets % n:
            return
        aug = n_sets // n
        last = n if max_games is None else min(n, max_games)
        for gid in range(0, last, stride):
            yield (f[f"boards_{gid * aug}"][:],
                   f[f"policies_{gid * aug}"][:],
                   f[f"values_{gid * aug}"][:])


def collect_ground_truth(path, stride=1, max_games=None, inverted=False):
    """WIN-IN-1 and MUST-BLOCK sets from one generation's self play file.

    win-in-1  : mover can complete 4-in-a-row now  -> true value +1, and the
                policy argmax must be a winning column.
    must-block: mover has NO win, opponent has EXACTLY ONE winning column ->
                that column is forced.

    inverted=True reproduces tmp/ab_heads.py's mover rule, for contrast only.
    """
    win = {"X": [], "cols": [], "mask": [], "ply": []}
    blk = {"X": [], "col": [], "mask": [], "ply": []}
    for boards, _, _ in iter_states(path, stride, max_games):
        for st in boards:
            board = st[..., -1]
            if inverted:
                mover = 1 if int((board == 1).sum()) <= int((board == -1).sum()) else -1
            else:
                mover = mover_of(st)
            mine = win_cols(board, mover)
            if mine:
                win["X"].append(st)
                win["cols"].append(mine)
                win["mask"].append(legal_mask(board))
                win["ply"].append(ply_of(board))
                continue
            theirs = win_cols(board, -mover)
            if len(theirs) == 1:
                blk["X"].append(st)
                blk["col"].append(theirs[0])
                blk["mask"].append(legal_mask(board))
                blk["ply"].append(ply_of(board))
    return win, blk


def dedup_average(states, values):
    """Byte-identical states -> one sample, occurrence-mean target, plus the
    occurrence count. Mirrors Dataloader.dedup_average (value side only) so a
    MSE computed here is on the same objective as the training val_value_loss."""
    states = np.asarray(states)
    values = np.asarray(values, dtype=np.float64).reshape(len(states), -1)
    flat = states.reshape(len(states), -1)
    uniq, inverse, counts = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
    means = np.zeros((len(uniq), values.shape[1]))
    np.add.at(means, inverse, values)
    means /= counts[:, None]
    w = counts.astype(np.float64)
    w *= len(w) / max(1.0, w.sum())          # normalize to mean 1.0, as training does
    return uniq.reshape(-1, *states.shape[1:]), means.astype(np.float32), w


def run_model(sess, X, batch=256):
    P, V = [], []
    X = np.asarray(X, dtype=np.float32)
    for i in range(0, len(X), batch):
        p, v = sess.run(None, {"inputs": X[i:i + batch]})
        P.append(p)
        V.append(np.asarray(v).reshape(-1))
    if not P:
        return np.zeros((0, COLS)), np.zeros(0)
    return np.concatenate(P), np.concatenate(V)


def session(onnx_path, threads=1):
    import onnxruntime as rt
    rt.set_default_logger_severity(4)
    so = rt.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return rt.InferenceSession(onnx_path, sess_options=so,
                               providers=["CPUExecutionProvider"])
