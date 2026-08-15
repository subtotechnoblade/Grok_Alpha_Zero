"""Benchmark data harness.

Replicates Dataloader.Create_Train_Test_Split + dedup_average EXACTLY so the
numbers here are comparable to production val_loss.

Production semantics being matched (Dataloader.py:95-192):
  * window = the 4 generations ending at `gen` (files[-n_prev-1:] then [1:])
  * iterated NEWEST FIRST, train_percent 1.0 -0.2 per file, test_percent 0.1 -0.02
  * split BY GAME inside each file; all `aug` copies of a game follow it
  * dedup_average applied per split side AFTER pooling the window
  * sample weight = occurrence count renormalised to MEAN 1.0 per side

Extras this harness adds (production does not need them):
  * pure-z target recovered per ply and dedup-averaged over the SAME groups
  * occurrence counts kept so we can slice by "seen >= k times"
  * an optional fully-unseen NEXT generation as a second holdout
"""
import os
import numpy as np
import h5py as h5

# Derived from this file's location, not hardcoded: the same scripts run from
# Windows and from WSL, where an absolute C:/ path does not resolve.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOTS = {
    "run2": os.path.join(_REPO, "tmp", "run2_archive_gen0-19_oldvaluehead"),
    "run3": os.path.join(_REPO, "tmp", "run3_archive_gen0-15_bottleneck_preweightfix"),
}
# measured in probe2.py: max|value| inside DRAWN games saturates at q_weight
Q_WEIGHT = {"run2": 0.2, "run3": 0.4}

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE, exist_ok=True)


def _z_from_values(v):
    """Recover the pure game outcome z, from the perspective of the player to move.

    Self_Play.py:162  target = (1-qw)*z + qw*q.  The final recorded ply is either
    the winning move (z=+1, q=+1 -> value exactly 1.0) or the board-filling drawn
    move (z=0, q=0 -> value exactly 0.0).  z then alternates sign backwards.
    """
    T = len(v)
    z_last = float(np.round(v[-1]))
    return z_last * ((-1.0) ** (np.arange(T - 1, -1, -1)))


def _read_gen(root, gen):
    """-> list of per-game dicts {boards, policies, values, z} incl. augmentations."""
    path = f"{root}/{gen}/Self_Play_Data.h5"
    games = []
    try:
        f = h5.File(path, mode="r")
    except OSError as e:
        print(f"  [warn] gen {gen}: OSError {e}")
        return games
    with f:
        keys = set(f.keys())
        n_sets = (len(keys) - 1) // 3
        n_games = int(np.array(f["game_stats"])[2])
        if n_games <= 0 or n_sets % n_games != 0:
            return games
        aug = n_sets // n_games
        for g in range(n_games):
            b, p, vv, zz = [], [], [], []
            ok = True
            for d in range(g * aug, g * aug + aug):
                if f"boards_{d}" not in keys:
                    ok = False
                    break
                bo = np.array(f[f"boards_{d}"], dtype=np.int8)
                po = np.array(f[f"policies_{d}"], dtype=np.float32)
                va = np.array(f[f"values_{d}"], dtype=np.float64).reshape(len(bo), -1)
                if len(bo) == 0:
                    ok = False
                    break
                b.append(bo); p.append(po); vv.append(va)
                zz.append(_z_from_values(va[:, 0]).reshape(-1, 1))
            if not ok:
                continue
            games.append(dict(boards=np.concatenate(b), policies=np.concatenate(p),
                              values=np.concatenate(vv), z=np.concatenate(zz)))
    return games


def dedup_average(states, policies, values, extra_targets=()):
    """Byte-identical merge, occurrence-weighted means. Mirrors Dataloader.dedup_average
    (line 12-63) exactly; `extra_targets` gets the same group-mean treatment."""
    states = np.asarray(states)
    policies = np.asarray(policies, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(len(states), -1)

    flat = states.reshape(len(states), -1)
    uniq, inverse, counts = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()

    def gmean(arr):
        arr = np.asarray(arr, dtype=np.float64).reshape(len(states), -1)
        out = np.zeros((len(uniq), arr.shape[1]))
        np.add.at(out, inverse, arr)
        return out / counts[:, None]

    return (uniq.reshape(-1, *states.shape[1:]),
            gmean(policies).astype(np.float32),
            gmean(values).astype(np.float32),
            counts.astype(np.float32),
            [gmean(e).astype(np.float32) for e in extra_targets])


def build(run, gen, n_prev=4, train_percent=1.0, train_decay=0.2,
          test_percent=0.1, test_decay=0.02, next_gen_holdout=True, seed=0, verbose=True):
    """Return dict with train / val(in-window, production protocol) / nxt(next generation)."""
    root = ROOTS[run]
    rng = np.random.RandomState(seed)

    files = list(range(gen - n_prev, gen + 1))          # 5 gens
    files = files[1:]                                   # production drops the oldest
    files = files[::-1]                                 # newest first
    files = [g for g in files if g >= 0]

    tr_s, tr_p, tr_v, tr_z = [], [], [], []
    te_s, te_p, te_v, te_z = [], [], [], []
    tp, sp = train_percent, test_percent
    for gid in files:
        games = _read_gen(root, gid)
        n_games = len(games)
        order = rng.permutation(n_games)
        n_test = int(sp * n_games)
        test_ids = set(order[:n_test].tolist())
        rest = order[n_test:]
        train_ids = set(rest[:int(tp * len(rest))].tolist())
        n_tr = n_te = 0
        for g in range(n_games):
            if g in test_ids:
                S, P, V, Z = te_s, te_p, te_v, te_z; n_te += 1
            elif g in train_ids:
                S, P, V, Z = tr_s, tr_p, tr_v, tr_z; n_tr += 1
            else:
                continue
            S.append(games[g]["boards"]); P.append(games[g]["policies"])
            V.append(games[g]["values"]); Z.append(games[g]["z"])
        if verbose:
            print(f"  gen {gid}: {n_games} games -> train {n_tr} (tp={tp:.2f}) test {n_te} (sp={sp:.2f})")
        tp = max(tp - train_decay, 0.0)
        sp = max(sp - test_decay, 0.0)

    def pack(S, P, V, Z, tag):
        S = np.concatenate(S); P = np.concatenate(P); V = np.concatenate(V); Z = np.concatenate(Z)
        n0 = len(S)
        us, up, uv, cnt, (uz,) = dedup_average(S, P, V, extra_targets=(Z,))
        w = cnt * (len(cnt) / max(1.0, cnt.sum()))
        if verbose:
            print(f"  {tag}: {n0} -> {len(us)} unique ({100*(1-len(us)/n0):.1f}% removed), "
                  f"w mean {w.mean():.3f} max {w.max():.1f}")
        return dict(X=us, policy=up, y_blend=uv[:, 0].astype(np.float64),
                    y_z=uz[:, 0].astype(np.float64), w=w.astype(np.float64),
                    count=cnt.astype(np.int64), n_raw=n0)

    out = dict(train=pack(tr_s, tr_p, tr_v, tr_z, "train"),
               val=pack(te_s, te_p, te_v, te_z, "val(in-window)"),
               run=run, gen=gen, q_weight=Q_WEIGHT[run])

    if next_gen_holdout:
        games = _read_gen(root, gen + 1)
        if games:
            S = np.concatenate([g["boards"] for g in games])
            P = np.concatenate([g["policies"] for g in games])
            V = np.concatenate([g["values"] for g in games])
            Z = np.concatenate([g["z"] for g in games])
            us, up, uv, cnt, (uz,) = dedup_average(S, P, V, extra_targets=(Z,))
            w = cnt * (len(cnt) / max(1.0, cnt.sum()))
            if verbose:
                print(f"  nxt(gen {gen+1}): {len(S)} -> {len(us)} unique")
            out["nxt"] = dict(X=us, policy=up, y_blend=uv[:, 0].astype(np.float64),
                              y_z=uz[:, 0].astype(np.float64), w=w.astype(np.float64),
                              count=cnt.astype(np.int64), n_raw=len(S))
    return out


def get(run, gen, seed=0, **kw):
    """Cached build()."""
    path = os.path.join(CACHE, f"{run}_g{gen}_s{seed}.npz")
    if os.path.exists(path):
        d = np.load(path, allow_pickle=False)
        out = {"run": run, "gen": gen, "q_weight": Q_WEIGHT[run]}
        for split in ("train", "val", "nxt"):
            if f"{split}_X" in d:
                out[split] = {k: d[f"{split}_{k}"] for k in
                              ("X", "policy", "y_blend", "y_z", "w", "count")}
        return out
    out = build(run, gen, seed=seed, **kw)
    flat = {}
    for split in ("train", "val", "nxt"):
        if split in out:
            for k in ("X", "policy", "y_blend", "y_z", "w", "count"):
                flat[f"{split}_{k}"] = out[split][k]
    np.savez_compressed(path, **flat)
    return out


def raw_window(run, gen, n_prev=4, train_percent=1.0, train_decay=0.2,
               test_percent=0.1, test_decay=0.02, seed=0):
    """Same window/split as build(), but the TRAIN side is returned as a list of
    per-game arrays (pre-dedup) so a learning curve can subsample by GAME, which is
    what "more self play" actually buys. val/nxt come back packed as usual."""
    root = ROOTS[run]
    rng = np.random.RandomState(seed)
    files = [g for g in list(range(gen - n_prev, gen + 1))[1:][::-1] if g >= 0]

    train_games = []
    te_s, te_p, te_v, te_z = [], [], [], []
    tp, sp = train_percent, test_percent
    for gid in files:
        games = _read_gen(root, gid)
        order = rng.permutation(len(games))
        n_test = int(sp * len(games))
        test_ids = set(order[:n_test].tolist())
        rest = order[n_test:]
        train_ids = set(rest[:int(tp * len(rest))].tolist())
        for g in range(len(games)):
            if g in test_ids:
                te_s.append(games[g]["boards"]); te_p.append(games[g]["policies"])
                te_v.append(games[g]["values"]); te_z.append(games[g]["z"])
            elif g in train_ids:
                train_games.append(games[g])
        tp = max(tp - train_decay, 0.0)
        sp = max(sp - test_decay, 0.0)

    def pack(S, P, V, Z):
        S = np.concatenate(S); P = np.concatenate(P); V = np.concatenate(V); Z = np.concatenate(Z)
        us, up, uv, cnt, (uz,) = dedup_average(S, P, V, extra_targets=(Z,))
        w = cnt * (len(cnt) / max(1.0, cnt.sum()))
        return dict(X=us, policy=up, y_blend=uv[:, 0].astype(np.float64),
                    y_z=uz[:, 0].astype(np.float64), w=w.astype(np.float64),
                    count=cnt.astype(np.int64), n_raw=len(S))

    val_games = [dict(boards=b, policies=p, values=v, z=z)
                 for b, p, v, z in zip(te_s, te_p, te_v, te_z)]
    out = dict(train_games=train_games, val_games=val_games,
               val=pack(te_s, te_p, te_v, te_z))
    games = _read_gen(root, gen + 1)
    if games:
        out["nxt_games"] = games
        out["nxt"] = pack([g["boards"] for g in games], [g["policies"] for g in games],
                          [g["values"] for g in games], [g["z"] for g in games])
    return out


def group_stats(games, target="z"):
    """Per unique state: occurrence count, mean target, and the WITHIN-group variance
    of the raw per-occurrence targets. The variance is the label noise that dedup
    averaging is fighting; sigma^2/n is the irreducible MSE against the averaged label."""
    S = np.concatenate([g["boards"] for g in games])
    key = "z" if target == "z" else "values"
    V = np.concatenate([g[key] for g in games]).reshape(len(S), -1)[:, 0].astype(np.float64)
    flat = S.reshape(len(S), -1)
    uniq, inverse, counts = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    s1 = np.zeros(len(uniq)); s2 = np.zeros(len(uniq))
    np.add.at(s1, inverse, V)
    np.add.at(s2, inverse, V * V)
    mean = s1 / counts
    # unbiased within-group variance; 0 (not nan) for singletons - flagged by count==1
    var = np.where(counts > 1, (s2 - counts * mean ** 2) / np.maximum(counts - 1, 1), np.nan)
    return dict(X=uniq.reshape(-1, *S.shape[1:]), count=counts.astype(np.int64),
                mean=mean, var=var,
                w=(counts * (len(counts) / counts.sum())).astype(np.float64))


def pack_games(games):
    """dedup_average a list of per-game dicts into a packed train pool."""
    S = np.concatenate([g["boards"] for g in games])
    P = np.concatenate([g["policies"] for g in games])
    V = np.concatenate([g["values"] for g in games])
    Z = np.concatenate([g["z"] for g in games])
    us, up, uv, cnt, (uz,) = dedup_average(S, P, V, extra_targets=(Z,))
    w = cnt * (len(cnt) / max(1.0, cnt.sum()))
    return dict(X=us, policy=up, y_blend=uv[:, 0].astype(np.float64),
                y_z=uz[:, 0].astype(np.float64), w=w.astype(np.float64),
                count=cnt.astype(np.int64), n_raw=len(S))


# ---------------------------------------------------------------- weighted MSE
def wmse(y, pred, w):
    y = np.asarray(y, dtype=np.float64).ravel()
    pred = np.asarray(pred, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    return float(np.sum(w * (y - pred) ** 2) / np.sum(w))


if __name__ == "__main__":
    import sys
    run = sys.argv[1] if len(sys.argv) > 1 else "run2"
    gen = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    # sanity: last-ply value must be 0 or 1 for z recovery to be valid
    root = ROOTS[run]
    bad, tot = 0, 0
    for g in range(gen - 3, gen + 2):
        try:
            with h5.File(f"{root}/{g}/Self_Play_Data.h5", "r") as f:
                keys = set(f.keys())
                n_sets = (len(keys) - 1) // 3
                for i in range(0, n_sets, 2):
                    if f"values_{i}" not in keys:
                        continue
                    v = float(np.array(f[f"values_{i}"]).ravel()[-1])
                    tot += 1
                    if not (abs(v - 1.0) < 1e-5 or abs(v) < 1e-5):
                        bad += 1
        except OSError:
            pass
    print(f"z-recovery sanity: last-ply value not in {{0,1}} for {bad}/{tot} games")
    d = build(run, gen)
    for s in ("train", "val", "nxt"):
        if s in d:
            p = d[s]
            print(f"{s}: n={len(p['X'])} raw={p['n_raw']} "
                  f"blend[min={p['y_blend'].min():.3f} max={p['y_blend'].max():.3f} "
                  f"mean={np.average(p['y_blend'], weights=p['w']):.4f} "
                  f"var_w={np.average((p['y_blend']-np.average(p['y_blend'],weights=p['w']))**2, weights=p['w']):.4f}] "
                  f"z[mean={np.average(p['y_z'], weights=p['w']):.4f} "
                  f"var_w={np.average((p['y_z']-np.average(p['y_z'],weights=p['w']))**2, weights=p['w']):.4f}] "
                  f"count: once={np.mean(p['count']==1)*100:.1f}% ge10={np.mean(p['count']>=10)*100:.1f}%")
