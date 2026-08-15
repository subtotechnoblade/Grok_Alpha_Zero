"""Shared loading / z-reconstruction for the value-target audit.

Everything here is read-only against the two STATIC archives.
"""
import os
import numpy as np
import h5py as h5

RUNS = {
    "run2": r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero\tmp\run2_archive_gen0-19_oldvaluehead",
    "run3": r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero\tmp\run3_archive_gen0-15_bottleneck_preweightfix",
}

# ply buckets, inclusive lower / inclusive upper
BUCKETS = [("0-9", 0, 9), ("10-21", 10, 21), ("22-41", 22, 41)]


def bucket_of(ply):
    if ply <= 9:
        return 0
    if ply <= 21:
        return 1
    return 2


# ---------------------------------------------------------------- win check
def _wins(board, col, player):
    """Drop `player` into `col` of (6,7) board (row 5 == bottom); True if it makes 4."""
    filled = np.count_nonzero(board[:, col])
    if filled >= 6:
        return None  # illegal
    r = 5 - filled
    b = board.copy()
    b[r, col] = player
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        run = 1
        for s in (1, -1):
            rr, cc = r + dr * s, col + dc * s
            while 0 <= rr < 6 and 0 <= cc < 7 and b[rr, cc] == player:
                run += 1
                rr += dr * s
                cc += dc * s
        if run >= 4:
            return True
    return False


def game_is_decided(last_board, mover):
    """Last stored ply == position the (possibly) winning move was played FROM."""
    for c in range(7):
        w = _wins(last_board, c, mover)
        if w:
            return True
    return False


# ---------------------------------------------------------------- loading
def open_gen(run, gen):
    path = os.path.join(RUNS[run], str(gen), "Self_Play_Data.h5")
    return h5.File(path, mode="r")


def gen_meta(f):
    keys = list(f.keys())
    n_sets = (len(keys) - 1) // 3
    gs = np.array(f["game_stats"])
    n_games = int(gs[2])
    aug = n_sets // n_games if n_games else 0
    return n_sets, n_games, aug, gs


def load_games(run, gen, max_games=None, with_mirrors=False):
    """Return list of dicts: boards (L,6,7,4) int8, target (L,), mover (L,), z (L,), decided.

    z reconstruction: alternate back from +1 at the last stored ply for decided
    games, 0 throughout for draws.  Draw/decided is decided by an EXACT win test
    on the last stored board, not by any property of the targets.
    """
    out = []
    with open_gen(run, gen) as f:
        n_sets, n_games, aug, gs = gen_meta(f)
        ng = n_games if max_games is None else min(max_games, n_games)
        for g in range(ng):
            copies = range(g * aug, g * aug + aug) if with_mirrors else (g * aug,)
            base = None
            for k, ds in enumerate(copies):
                b = np.array(f[f"boards_{ds}"])
                v = np.array(f[f"values_{ds}"]).reshape(-1)
                if k == 0:
                    L = len(v)
                    mover = -b[:, 0, 0, 0].astype(np.int8)
                    dec = game_is_decided(b[-1, :, :, 3].astype(np.int8), int(mover[-1]))
                    if dec:
                        z = np.empty(L, dtype=np.float32)
                        z[::-1] = np.where(np.arange(L) % 2 == 0, 1.0, -1.0)
                    else:
                        z = np.zeros(L, dtype=np.float32)
                    base = dict(z=z, decided=dec, mover=mover, L=L, game=g, gen=gen, run=run)
                out.append(dict(base, boards=b, target=v, aug=k))
    return out


def infer_q_weight(games, candidates=(0.0, 0.05, 0.1, 0.12, 0.15, 0.18, 0.2,
                                      0.25, 0.3, 0.35, 0.4, 0.45, 0.5)):
    """Smallest q_weight for which q = (target-(1-qw)z)/qw lies fully in [-1,1].

    Assuming a LARGER qw than the truth can never violate the bound, so the
    smallest consistent candidate is the truth.  Returns (qw, table).
    """
    z = np.concatenate([g["z"] for g in games])
    t = np.concatenate([g["target"] for g in games])
    table = []
    best = None
    for qw in candidates:
        if qw == 0.0:
            frac = float(np.mean(np.abs(t - z) < 1e-5))
            mx = float(np.abs(t - z).max())
            table.append((qw, frac, mx))
            if frac == 1.0 and best is None:
                best = qw
            continue
        q = (t - (1.0 - qw) * z) / qw
        frac = float(np.mean(np.abs(q) <= 1.0 + 1e-4))
        table.append((qw, frac, float(np.abs(q).max())))
        if frac == 1.0 and best is None:
            best = qw
    return best, table


def stack_samples(games, key_mode="state"):
    """Flatten games -> arrays. key = bytes of the exact net input (or board+mover)."""
    keys, z, tgt, ply, gid = [], [], [], [], []
    for g in games:
        b = g["boards"]
        L = len(g["target"])
        for i in range(L):
            if key_mode == "state":
                keys.append(b[i].tobytes())
            else:  # board plane + side to move only
                keys.append(b[i, :, :, 3].tobytes() + bytes([int(g["mover"][i]) & 0xFF]))
        z.append(g["z"])
        tgt.append(g["target"])
        ply.append(np.arange(L, dtype=np.int32))
        gid.append(np.full(L, g["game"], dtype=np.int32))
    return (np.array(keys, dtype=object),
            np.concatenate(z).astype(np.float64),
            np.concatenate(tgt).astype(np.float64),
            np.concatenate(ply),
            np.concatenate(gid))


def group_index(keys):
    """keys (object array of bytes) -> (inverse, counts, n_unique) via dict."""
    d = {}
    inv = np.empty(len(keys), dtype=np.int64)
    for i, k in enumerate(keys):
        j = d.get(k)
        if j is None:
            j = len(d)
            d[k] = j
        inv[i] = j
    counts = np.bincount(inv, minlength=len(d))
    return inv, counts, len(d), d
