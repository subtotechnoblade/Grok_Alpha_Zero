"""Feature views of a Connect4 input state.

Input state layout (Connect4/game.py:267-286): (6, 7, 4) int8
  ch0 = constant plane holding the player who JUST MOVED (+-1)
  ch1 = board 2 plies ago, ch2 = board 1 ply ago, ch3 = current board
  board cells hold the ABSOLUTE owner id (+-1), row 0 = top, row 5 = bottom.

VERIFIED, not assumed (see the ch0 check in the session log): game.get_input_state
passes -next_player, and MCTS.py:461/488 uses the same convention, so the player
TO MOVE is -ch0. Self_Play.py:110-156 makes the value target the outcome for the
player to move. So mover-relative views multiply the board planes by -ch0.
Empirically: count(cells == ch0) == ceil(ply/2) for 100.0% of 44212 states.
"""
import numpy as np

ROWS, COLS = 6, 7

# every 4-in-a-row window, as flat indices into a (6,7) board
def _lines():
    out = []
    for r in range(ROWS):
        for c in range(COLS):
            if c + 3 < COLS:
                out.append([r * COLS + c + i for i in range(4)])
            if r + 3 < ROWS:
                out.append([(r + i) * COLS + c for i in range(4)])
            if c + 3 < COLS and r + 3 < ROWS:
                out.append([(r + i) * COLS + c + i for i in range(4)])
            if c - 3 >= 0 and r + 3 < ROWS:
                out.append([(r + i) * COLS + c - i for i in range(4)])
    return np.array(out, dtype=np.int64)

LINES = _lines()                      # (69, 4)
LINE_ROWS = LINES // COLS             # row of each cell in each line


def raw(X):
    """Flattened planes, exactly as the network sees them. (N, 168)"""
    return np.asarray(X, dtype=np.float32).reshape(len(X), -1)


def rel(X):
    """Mover-relative planes + mover id. (N, 127)"""
    X = np.asarray(X, dtype=np.float32)
    mover = -X[:, 0, 0, 0]                                 # player TO MOVE = -ch0
    planes = X[:, :, :, 1:4] * mover[:, None, None, None]  # +1 = mine
    return np.concatenate([planes.reshape(len(X), -1), mover[:, None]], axis=1).astype(np.float32)


HAND_NAMES = (
    [f"my_col{c}" for c in range(COLS)] + [f"opp_col{c}" for c in range(COLS)] +
    ["my_open3", "opp_open3", "my_playable3", "opp_playable3",
     "my_open2", "opp_open2", "my_open3_oddrow", "opp_open3_oddrow",
     "ply", "side_to_move", "my_centre", "opp_centre",
     "my_centre_weighted", "opp_centre_weighted", "my_total", "opp_total"]
)


def hand(X):
    """~26 hand-engineered, mover-relative features. (N, 26)"""
    X = np.asarray(X, dtype=np.int8)
    N = len(X)
    mover = -X[:, 0, 0, 0].astype(np.int32)                # player TO MOVE = -ch0
    board = X[:, :, :, 3].astype(np.int32) * mover[:, None, None]   # +1 mine, -1 opp
    flat = board.reshape(N, ROWS * COLS)

    mine = (flat == 1)
    opp = (flat == -1)
    empty = (flat == 0)

    occ = (flat != 0).reshape(N, ROWS, COLS)
    col_fill = occ.sum(1)                                  # (N,7) discs per column
    my_col = mine.reshape(N, ROWS, COLS).sum(1).astype(np.float32)
    opp_col = opp.reshape(N, ROWS, COLS).sum(1).astype(np.float32)

    # drop row per column (row index the next disc lands on); 6 == column full
    drop_row = 5 - col_fill                                # (N,7)
    playable = np.zeros((N, ROWS * COLS), dtype=bool)
    ok = drop_row >= 0
    rr = np.clip(drop_row, 0, 5)
    idx = rr * COLS + np.arange(COLS)[None, :]
    np.put_along_axis(playable, idx, ok, axis=1)

    L = LINES.ravel()
    my_l = mine[:, L].reshape(N, -1, 4).sum(2)             # (N,69)
    op_l = opp[:, L].reshape(N, -1, 4).sum(2)
    em_l = empty[:, L].reshape(N, -1, 4).sum(2)
    pl_l = playable[:, L].reshape(N, -1, 4)
    # row of the (single) empty cell in a 3+1 line, for parity
    emp_cell = empty[:, L].reshape(N, -1, 4)
    row_of_empty = (LINE_ROWS[None, :, :] * emp_cell).sum(2)

    my3 = (my_l == 3) & (em_l == 1)
    op3 = (op_l == 3) & (em_l == 1)
    my3_playable = (my3 & (emp_cell & pl_l).any(2))
    op3_playable = (op3 & (emp_cell & pl_l).any(2))
    my2 = ((my_l == 2) & (em_l == 2)).sum(1)
    op2 = ((op_l == 2) & (em_l == 2)).sum(1)
    # odd rows (5,3,1 = 1st/3rd/5th from bottom) are the classic parity edge
    odd = ((5 - row_of_empty) % 2 == 0)
    my3_odd = (my3 & odd).sum(1)
    op3_odd = (op3 & odd).sum(1)

    centre_w = np.abs(np.arange(COLS) - 3.0)
    cw = (3.0 - centre_w)[None, :]

    feats = [
        my_col, opp_col,
        my3.sum(1)[:, None], op3.sum(1)[:, None],
        my3_playable.sum(1)[:, None], op3_playable.sum(1)[:, None],
        my2[:, None], op2[:, None],
        my3_odd[:, None], op3_odd[:, None],
        occ.sum((1, 2))[:, None],                          # ply
        mover[:, None],                                    # absolute side to move
        my_col[:, 3:4], opp_col[:, 3:4],
        (my_col * cw).sum(1)[:, None], (opp_col * cw).sum(1)[:, None],
        my_col.sum(1)[:, None], opp_col.sum(1)[:, None],
    ]
    return np.concatenate([np.asarray(f, dtype=np.float32).reshape(N, -1) for f in feats], 1)


def hand_rel(X):
    return np.concatenate([hand(X), rel(X)], axis=1).astype(np.float32)


VIEWS = {"raw": raw, "rel": rel, "hand": hand, "hand+rel": hand_rel}

if __name__ == "__main__":
    import data
    d = data.get("run2", 12)
    X = d["train"]["X"][:5]
    for n, fn in VIEWS.items():
        print(n, fn(X).shape)
    print(len(HAND_NAMES), hand(X).shape)
    print(dict(zip(HAND_NAMES, hand(X)[3].tolist())))
