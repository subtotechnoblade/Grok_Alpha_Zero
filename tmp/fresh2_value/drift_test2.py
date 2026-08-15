"""Value-head divergence: drift (A) vs variance floor (B) vs staleness (C).

Window = gens 8-11 (latest 4 COMPLETED), net = gen 12 model.onnx (trained on
exactly that window: Create_Train_Test_Split takes the last 5 dirs, drops the
oldest, ramp 1.0/0.9/0.8/0.7 newest-first).

Fixes vs drift_test.py:
- common keys = union over gens (old version only looked at keys of the OLDEST gen)
- between-gen variance is also reported NOISE-CORRECTED: raw var(per-gen means)
  contains sampling noise of each mean (~within_var/n_g); subtract it to get the
  true regime drift.
- weight audit replicates the real training pool: BOTH mirror augmentations,
  recency ramp, test split removed, dedup counts -> weights normalized to mean 1.

Read-only h5 access (mode="r"), OSError-guarded. CPU only.
"""
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import h5py as h5
import onnxruntime as rt

rt.set_default_logger_severity(4)

GENS = [8, 9, 10, 11]          # oldest -> newest, all COMPLETED (12 is mid-write)
NET = "Connect4/Grok_Zero_Train/12/model.onnx"
PLY_BUCKETS = [(0, 5), (6, 15), (16, 41)]
# training ramp, newest gen first (Create_Train_Test_Split iterates reversed)
TRAIN_PCT = {11: 1.0, 10: 0.9, 9: 0.8, 8: 0.7}
TEST_PCT = {11: 0.10, 10: 0.09, 9: 0.08, 8: 0.07}


def load_gen(gen, mirrored=False):
    """Return (games) list; each game = (boards[np], values[np]) un-mirrored copy,
    or both augmentation copies if mirrored=True (list of per-dataset tuples,
    grouped per game)."""
    path = f"Connect4/Grok_Zero_Train/{gen}/Self_Play_Data.h5"
    games = []
    try:
        with h5.File(path, "r") as f:
            n = int(f["game_stats"][2])
            n_sets = (len(f.keys()) - 1) // 3
            aug = n_sets // n
            for gid in range(n):
                copies = []
                rng = range(gid * aug, gid * aug + aug) if mirrored else [gid * aug]
                for did in rng:
                    copies.append((f[f"boards_{did}"][:], f[f"values_{did}"][:]))
                games.append(copies)
    except OSError as e:
        print(f"  !! gen {gen} busy/unreadable ({e}), skipped")
    return games


def main():
    # ---------- pass 1: per-gen per-position targets (un-mirrored) ----------
    per_gen = {}                     # gen -> {key: np.array of targets}
    ply_of = {}
    gen_games = {}
    for g in GENS:
        games = load_gen(g, mirrored=False)
        gen_games[g] = games
        d = defaultdict(list)
        for copies in games:
            boards, values = copies[0]
            for ply, (b, v) in enumerate(zip(boards, values)):
                d[b.tobytes()].append(float(v[0]))
                ply_of.setdefault(b.tobytes(), ply)
        per_gen[g] = {k: np.asarray(v) for k, v in d.items()}
        n_once = sum(1 for v in per_gen[g].values() if len(v) == 1)
        n_samp = sum(len(v) for v in per_gen[g].values())
        print(f"gen {g}: {n_samp:>6} samples, {len(per_gen[g]):>6} unique, "
              f"{n_once / len(per_gen[g]):.1%} occur once")

    # pooled window occurrence structure (what dedup actually sees, un-mirrored proxy)
    pooled_counts = defaultdict(int)
    for g in GENS:
        for k, v in per_gen[g].items():
            pooled_counts[k] += len(v)
    total_u = len(pooled_counts)
    total_s = sum(pooled_counts.values())
    once = sum(1 for c in pooled_counts.values() if c == 1)
    print(f"\nPOOLED window (un-mirrored): {total_s} samples -> {total_u} unique; "
          f"{once/total_u:.1%} of unique positions occur ONCE "
          f"({once/total_s:.1%} of raw sample mass, and after mean-1 weight "
          f"normalization they hold {once/total_u:.1%} of total loss weight)")

    # ---------- (A) vs (B): between-gen drift vs within-gen variance ----------
    print(f"\n(A) DRIFT vs (B) WITHIN-GEN VARIANCE - positions in >=3 of gens {GENS}")
    all_keys = set()
    for g in GENS:
        all_keys |= per_gen[g].keys()
    common = [k for k in all_keys if sum(k in per_gen[g] for g in GENS) >= 3]
    print(f"{'ply':>6} {'n pos':>7} {'between(raw)':>13} {'noise':>8} "
          f"{'between(corr)':>14} {'within':>8} {'raw ratio':>10} {'corr ratio':>11}")
    for lo, hi in PLY_BUCKETS:
        raw_b, corr_b, noise_l, within_l = [], [], [], []
        for k in common:
            if not (lo <= ply_of[k] <= hi):
                continue
            means, wvars, noises = [], [], []
            for g in GENS:
                if k in per_gen[g]:
                    vals = per_gen[g][k]
                    means.append(vals.mean())
                    if len(vals) >= 2:
                        wv = vals.var(ddof=1)
                        wvars.append(wv)
                        noises.append(wv / len(vals))
            if len(means) >= 3:
                bv = np.var(means, ddof=1)
                raw_b.append(bv)
                if wvars:
                    within_l.append(np.mean(wvars))
                    nz = np.mean(noises)
                    noise_l.append(nz)
                    corr_b.append(max(0.0, bv - nz))
        if raw_b:
            b = np.mean(raw_b)
            w = np.mean(within_l) if within_l else float("nan")
            nz = np.mean(noise_l) if noise_l else float("nan")
            cb = np.mean(corr_b) if corr_b else float("nan")
            print(f"{f'{lo}-{hi}':>6} {len(raw_b):>7} {b:>13.4f} {nz:>8.4f} "
                  f"{cb:>14.4f} {w:>8.4f} "
                  f"{b / w if w > 0 else float('nan'):>10.2f} "
                  f"{cb / w if w > 0 else float('nan'):>11.2f}")

    # ---------- WEIGHT AUDIT: replicate the real training pool ----------
    print("\nWEIGHT AUDIT - replicated training pool (mirrors + ramp + test removed, seed 0)")
    rng = np.random.default_rng(0)
    pool = defaultdict(lambda: {"n": 0, "sum": 0.0, "gens": defaultdict(list), "ply": -1})
    n_pool = 0
    for g in sorted(GENS, reverse=True):          # newest first, like training
        games = load_gen(g, mirrored=True)
        n = len(games)
        perm = rng.permutation(n)
        n_test = int(TEST_PCT[g] * n)
        rest = perm[n_test:]
        train_ids = set(rest[:int(TRAIN_PCT[g] * len(rest))].tolist())
        for gid in train_ids:
            for boards, values in games[gid]:
                for ply, (b, v) in enumerate(zip(boards, values)):
                    e = pool[b.tobytes()]
                    e["n"] += 1
                    e["sum"] += float(v[0])
                    e["gens"][g].append(float(v[0]))
                    e["ply"] = ply
                    n_pool += 1
    keys = list(pool.keys())
    counts = np.array([pool[k]["n"] for k in keys], dtype=np.float64)
    weights = counts * (len(counts) / counts.sum())    # mean 1.0, exactly like Dataloader
    plys = np.array([pool[k]["ply"] for k in keys])
    means = np.array([pool[k]["sum"] / pool[k]["n"] for k in keys])
    order = np.argsort(-weights)
    top10 = order[:10]
    tot_w = weights.sum()
    print(f"pool: {n_pool} samples -> {len(keys)} unique ({1 - len(keys)/n_pool:.1%} removed)")
    print(f"weights: mean 1.0, max {weights.max():.1f}, p99 {np.percentile(weights, 99):.2f}, "
          f"p99.9 {np.percentile(weights, 99.9):.1f}")
    print(f"top-10 samples hold {weights[top10].sum() / tot_w:.2%} of total loss weight; "
          f"max single = {weights.max() / tot_w:.2%} of all weight "
          f"= {weights.max() / 1024:.1%} of a 1024 batch if drawn")
    for lo, hi in [(0, 4)] + PLY_BUCKETS:
        m = (plys >= lo) & (plys <= hi)
        print(f"  plies {lo}-{hi}: {m.sum():>6} unique ({m.mean():.1%}), "
              f"{weights[m].sum() / tot_w:.2%} of total loss weight, "
              f"{counts[m].sum() / counts.sum():.2%} of raw samples")
    print("\ntop-10 heaviest positions (per-gen mean target = drift in its purest form):")
    print(f"{'rank':>4} {'ply':>4} {'count':>6} {'weight':>8} {'pooled tgt':>10} "
          + " ".join(f"g{g} mean(n)" for g in GENS))
    for r, i in enumerate(top10):
        k = keys[i]
        e = pool[k]
        gcols = []
        for g in GENS:
            vs = e["gens"].get(g)
            gcols.append(f"{np.mean(vs):+.3f}({len(vs)})" if vs else "     -    ")
        print(f"{r + 1:>4} {plys[i]:>4} {int(counts[i]):>6} {weights[i]:>8.1f} "
              f"{means[i]:>+10.3f}  " + " ".join(gcols))

    # ---------- (C) + per-ply error profile with the gen-12 net ----------
    print(f"\n(C) gen-12 net value MSE per source generation, ALL un-mirrored samples")
    sess = rt.InferenceSession(NET, providers=["CPUExecutionProvider"])

    def predict(boards):
        out = []
        for i in range(0, len(boards), 1024):
            _, val = sess.run(None, {"inputs": boards[i:i + 1024].astype(np.float32)})
            out.append(val[:, 0])
        return np.concatenate(out)

    preds_gen, targ_gen, ply_gen, key_gen = {}, {}, {}, {}
    for g in GENS:
        B, V, P, K = [], [], [], []
        for copies in gen_games[g]:
            boards, values = copies[0]
            for ply, (b, v) in enumerate(zip(boards, values)):
                B.append(b); V.append(float(v[0])); P.append(ply); K.append(b.tobytes())
        B = np.asarray(B); V = np.asarray(V); P = np.asarray(P)
        pr = predict(B)
        preds_gen[g], targ_gen[g], ply_gen[g], key_gen[g] = pr, V, P, K
        print(f"  gen {g}: MSE {np.mean((pr - V) ** 2):.4f}   (n={len(V)})")

    print(f"\nPER-PLY ERROR PROFILE - gen-12 net MSE vs RAW targets, per gen x ply bucket")
    hdr = f"{'bucket':>8}" + "".join(f"{f'gen {g}':>10}" for g in GENS)
    print(hdr)
    for lo, hi in PLY_BUCKETS:
        row = f"{f'{lo}-{hi}':>8}"
        for g in GENS:
            m = (ply_gen[g] >= lo) & (ply_gen[g] <= hi)
            row += f"{np.mean((preds_gen[g][m] - targ_gen[g][m]) ** 2):>10.4f}"
        print(row)

    # decomposition on the NEWEST gen: raw MSE = fit error vs per-pos mean + noise floor
    g = GENS[-1]
    posmean = {k: v.mean() for k, v in per_gen[g].items()}
    pm = np.array([posmean[k] for k in key_gen[g]])
    pooled_mean = {keys[i]: means[i] for i in range(len(keys))}
    print(f"\nDECOMPOSITION on newest gen {g}:  raw_MSE = (pred - gen{g} pos-mean)^2 "
          f"+ noise floor (target - pos-mean)^2;  drift penalty = (pooled window "
          f"target - gen{g} pos-mean)^2 [the error even a PERFECT fit of the "
          f"training target must make]")
    print(f"{'bucket':>8} {'raw MSE':>9} {'fit err':>9} {'noise floor':>12} "
          f"{'drift penalty':>14} {'n':>7}")
    for lo, hi in PLY_BUCKETS:
        m = (ply_gen[g] >= lo) & (ply_gen[g] <= hi)
        raw = np.mean((preds_gen[g][m] - targ_gen[g][m]) ** 2)
        fit = np.mean((preds_gen[g][m] - pm[m]) ** 2)
        floor = np.mean((targ_gen[g][m] - pm[m]) ** 2)
        has = np.array([k in pooled_mean for k in key_gen[g]]) & m
        dp = np.mean([(pooled_mean[key_gen[g][i]] - pm[i]) ** 2
                      for i in np.where(has)[0]]) if has.any() else float("nan")
        print(f"{f'{lo}-{hi}':>8} {raw:>9.4f} {fit:>9.4f} {floor:>12.4f} "
              f"{dp:>14.4f} {int(m.sum()):>7}")


if __name__ == "__main__":
    main()
