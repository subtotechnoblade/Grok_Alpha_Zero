"""Per-arm, per-generation ground-truth metrics for the value-head experiment.

    python tmp/final_exp/metrics.py <arm> [<arm> ...]

For each generation g of each arm (model g = the net that PLAYED generation g,
trained on generations max(0,g-4)..g-1, so generation g's positions are held out
for it by construction):

  FIXED BENCHMARK (bench.npz -- identical set for every arm, drawn from the
  archived runs, in no arm's training data):
    win-in-1  value : mean V, P(V>0), P(V>0.9), MSE vs the true value +1
                      (a constant-zero predictor scores MSE 1.000)
    win-in-1  policy: top-1 lands on a winning column
    must-block policy: top-1 is the single forced column
  OWN DATA (same three, on generation g's own positions -- on distribution)
  VAL vs NXT: value MSE on the in-window holdout (the pipeline's own game-level
    test split, read from the training log at the epoch ModelCheckpoint kept)
    versus value MSE on generation g's data, both on dedup-averaged targets with
    occurrence weights normalized to mean 1, i.e. the same objective.
  COVERAGE: fraction of generation g's unique positions that appear anywhere in
    the window model g was trained on, by ply band.
  SELF PLAY: outcome split and average game length.
"""
import os
import re
import sys
import json
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
os.chdir(ROOT)

import numpy as np
import h5py as h5

from gtlib import (collect_ground_truth, dedup_average, run_model, session,
                   ply_of, band_of, BAND_NAMES, iter_states)

NUM_PREV = 4          # train_config["num_previous_generations"]
OWN_STRIDE = 2        # subsample games when collecting own-data ground truth


# ----------------------------------------------------------------- train log
EPOCH_RE = re.compile(
    r"-\s*loss:\s*([0-9.eE+-]+).*?-\s*value_loss:\s*([0-9.eE+-]+)"
    r".*?-\s*val_loss:\s*([0-9.eE+-]+).*?-\s*val_value_loss:\s*([0-9.eE+-]+)")
GEN_RE = re.compile(r"Started training for generation:\s*(\d+)")


def parse_train_log(path):
    """{generation_trained_on: {...}} keyed by the generation whose self play
    ENDED the window; the resulting model lands in folder generation+1."""
    if not os.path.exists(path):
        return {}
    out, cur, rows = {}, None, []

    def flush():
        if cur is not None and rows:
            best = min(rows, key=lambda r: r[2])          # ModelCheckpoint(val_loss)
            out[cur] = {"epochs": len(rows),
                        "train_loss": best[0], "train_value_loss": best[1],
                        "val_loss": best[2], "val_value_loss": best[3],
                        "last_train_value_loss": rows[-1][1],
                        "last_val_value_loss": rows[-1][3]}

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    for line in re.split(r"[\r\n]+", text):
        m = GEN_RE.search(line)
        if m:
            flush()
            cur, rows = int(m.group(1)), []
            continue
        m = EPOCH_RE.search(line)
        if m and cur is not None:
            rows.append(tuple(float(x) for x in m.groups()))
    flush()
    return out


# ------------------------------------------------------------------ coverage
def positions_of(path, stride=1):
    """(board-key set, full-state-key set, per-band board keys) for one gen."""
    boards, states = set(), set()
    band_keys = [set() for _ in BAND_NAMES]
    for b, _, _ in iter_states(path, stride):
        for st in b:
            bk = st[..., -1].tobytes()
            boards.add(bk)
            states.add(st.tobytes())
            band_keys[band_of(ply_of(st[..., -1]))].add(bk)
    return boards, states, band_keys


def coverage(run, g):
    """Fraction of gen g's positions present in gens max(0,g-4)..g-1."""
    cur = f"{run}/{g}/Self_Play_Data.h5"
    if not os.path.exists(cur):
        return None
    win_gens = [x for x in range(max(0, g - NUM_PREV), g)
                if os.path.exists(f"{run}/{x}/Self_Play_Data.h5")]
    if not win_gens:
        return None
    cb, cs, cband = positions_of(cur, stride=2)
    wb, ws = set(), set()
    wband = [set() for _ in BAND_NAMES]
    for x in win_gens:
        b, s, bb = positions_of(f"{run}/{x}/Self_Play_Data.h5", stride=2)
        wb |= b
        ws |= s
        for i in range(len(BAND_NAMES)):
            wband[i] |= bb[i]
    out = {"window": win_gens,
           "board": len(cb & wb) / max(1, len(cb)),
           "state": len(cs & ws) / max(1, len(cs)),
           "n_positions": len(cb)}
    for i, name in enumerate(BAND_NAMES):
        n = len(cband[i])
        out[f"band_{name}"] = (len(cband[i] & wband[i]) / n) if n else float("nan")
        out[f"nband_{name}"] = n
    return out


# ------------------------------------------------------------------- scoring
def score_value(V, mask_true_plus_one=True):
    return {"mean_V": float(V.mean()),
            "frac_gt0": float(np.mean(V > 0)),
            "frac_gt09": float(np.mean(V > 0.9)),
            "mse_vs_plus1": float(np.mean((V - 1.0) ** 2))}


def score_policy_win(P, colmask, legal):
    P = P * legal
    s = P.sum(1, keepdims=True)
    P = np.divide(P, s, out=np.full_like(P, 1.0 / P.shape[1]), where=s > 0)
    top1 = np.array([bool(colmask[i][int(np.argmax(P[i]))]) for i in range(len(P))])
    mass = np.array([float(P[i][colmask[i]].sum()) for i in range(len(P))])
    return {"top1": float(top1.mean()), "mass": float(mass.mean())}


def score_policy_block(P, col, legal):
    P = P * legal
    s = P.sum(1, keepdims=True)
    P = np.divide(P, s, out=np.full_like(P, 1.0 / P.shape[1]), where=s > 0)
    top1 = (P.argmax(1) == col)
    return {"top1": float(top1.mean()), "mass": float(P[np.arange(len(P)), col].mean())}


def value_mse_on_gen(sess, path, stride=1):
    """MSE of the model against the stored value targets of one generation,
    dedup-averaged and occurrence-weighted exactly as training does."""
    X, Y = [], []
    for b, _, v in iter_states(path, stride):
        X.append(b)
        Y.append(np.asarray(v).reshape(len(b), -1))
    if not X:
        return None
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    Xu, Yu, w = dedup_average(X, Y)
    _, V = run_model(sess, Xu)
    err = (V - Yu.reshape(-1)) ** 2
    return {"mse_weighted": float(np.sum(err * w) / np.sum(w)),
            "mse_unweighted": float(err.mean()),
            "n_unique": int(len(Xu)),
            "target_var": float(np.var(Yu.reshape(-1))),
            "target_mean": float(np.mean(Yu.reshape(-1))),
            "pred_mean": float(V.mean()),
            "pred_std": float(V.std())}


def selfplay_stats(path):
    with h5.File(path, "r") as f:
        mx, tot, n, l, d, w = [int(x) for x in f["game_stats"][:]]
    if n == 0:
        return None
    return {"games": n, "avg_len": tot / n, "max_len": mx,
            "p_minus1": l / n, "draw": d / n, "p_plus1": w / n}


# ----------------------------------------------------------------------- main
def analyse(arm, bench, maxgen=999):
    run = f"tmp/final_exp/{arm}/Grok_Zero_Train"
    log = f"tmp/final_exp/{arm}.log"
    trainlog = parse_train_log(log)
    gens = sorted(int(os.path.basename(os.path.dirname(p)))
                  for p in glob.glob(f"{run}/*/model.onnx"))
    gens = [g for g in gens if g <= maxgen]

    rows = []
    for g in gens:
        data = f"{run}/{g}/Self_Play_Data.h5"
        if not os.path.exists(data):
            continue
        try:
            sp = selfplay_stats(data)
        except OSError:
            continue
        if sp is None or sp["games"] < 10:
            continue                       # generation still being written
        try:
            sess = session(f"{run}/{g}/model.onnx")
        except Exception as exc:
            print(f"  gen {g}: cannot load model ({exc})")
            continue

        row = {"gen": g, "selfplay": sp}

        # ---- fixed benchmark
        _, V = run_model(sess, bench["winX"])
        Pw, _ = run_model(sess, bench["winX"])
        Pb, _ = run_model(sess, bench["blkX"])
        row["fixed"] = {
            "value": score_value(V),
            "win_policy": score_policy_win(Pw, bench["winColMask"], bench["winMask"]),
            "block_policy": score_policy_block(Pb, bench["blkCol"], bench["blkMask"]),
            "n_win": int(len(bench["winX"])), "n_blk": int(len(bench["blkX"]))}

        # ---- own data
        try:
            w, b = collect_ground_truth(data, stride=OWN_STRIDE)
        except OSError:
            w, b = {"X": []}, {"X": []}
        if w["X"]:
            Po, Vo = run_model(sess, w["X"])
            cm = np.zeros((len(w["X"]), 7), dtype=bool)
            for i, cols in enumerate(w["cols"]):
                cm[i, cols] = True
            own = {"value": score_value(Vo),
                   "win_policy": score_policy_win(Po, cm, np.asarray(w["mask"])),
                   "n_win": len(w["X"])}
            if b["X"]:
                Pbo, _ = run_model(sess, b["X"])
                own["block_policy"] = score_policy_block(
                    Pbo, np.asarray(b["col"]), np.asarray(b["mask"]))
                own["n_blk"] = len(b["X"])
            row["own"] = own

        # ---- val vs nxt
        row["nxt"] = value_mse_on_gen(sess, data, stride=2)
        prev = f"{run}/{g - 1}/Self_Play_Data.h5"
        if os.path.exists(prev):
            # last generation INSIDE the window model g trained on: a fit number,
            # log-independent, so the same contrast can be run on the archives
            row["inwin"] = value_mse_on_gen(sess, prev, stride=2)
            if row["inwin"] and row["nxt"]:
                row["gap_fit"] = (row["nxt"]["mse_weighted"]
                                  / max(1e-9, row["inwin"]["mse_weighted"]))
        tl = trainlog.get(g - 1)
        row["train"] = tl
        if tl and row["nxt"]:
            row["gap"] = row["nxt"]["mse_weighted"] / max(1e-9, tl["val_value_loss"])

        # ---- coverage
        row["coverage"] = coverage(run, g)

        rows.append(row)
        print(f"  gen {g}: done", flush=True)
    return rows


def main():
    bz = np.load("tmp/final_exp/bench.npz")
    bench = {k: bz[k] for k in bz.files}
    bench["winX"] = bench["winX"].astype(np.float32)
    bench["blkX"] = bench["blkX"].astype(np.float32)

    out = {}
    for arm in sys.argv[1:]:
        print(f"=== {arm} ===", flush=True)
        out[arm] = analyse(arm, bench)
    with open("tmp/final_exp/metrics.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote tmp/final_exp/metrics.json")


if __name__ == "__main__":
    main()
