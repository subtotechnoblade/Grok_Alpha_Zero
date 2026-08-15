"""ITEMS 3,4,5: dedup_average math, provenance traceback to raw h5, split-by-game,
recency ramp, leakage, __getitem__ contract.

Deterministic: np.random.seed is patched so the library's bare np.random.seed()
calls seed a fixed value, making the game permutations reproducible. An
instrumented replay of the collection loop is validated against the real
Create_Train_Test_Split output before any provenance claim is made.
"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
REPO = r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero"
sys.path.insert(0, REPO)

import numpy as np
import h5py as h5
from pathlib import Path
from glob import glob

SEED = 12345
_real_seed = np.random.seed
np.random.seed = lambda *a, **k: _real_seed(SEED)

import Dataloader as DL

ARCHIVE = os.path.join(REPO, "tmp", "run3_archive_gen0-15_bottleneck_preweightfix")
NPG, TRP, TRD, TEP, TED = 4, 1.0, 0.2, 0.1, 0.02

# ------------------------------------------------------------------ real run
splitter = DL.Create_Train_Test_Split(ARCHIVE, NPG, TRP, TRD, TEP, TED,
                                      dedup_average=True)
print("files in window (iteration order):",
      [Path(f).parent.name for f in splitter.files])
(tr_s, tr_p, tr_v, tr_w), (te_s, te_p, te_v, te_w) = splitter.split()
print("REAL: train %d uniq, test %d uniq" % (len(tr_s), len(te_s)))
print("REAL: train weight mean %.6f sum %.2f max %.1f" %
      (tr_w.mean(), tr_w.sum(), tr_w.max()))
print("REAL: test  weight mean %.6f sum %.2f max %.1f" %
      (te_w.mean(), te_w.sum(), te_w.max()))

# ------------------------------------------------ instrumented replay (same seed)
files = splitter.files
raw = {"train": {"s": [], "p": [], "v": [], "prov": []},
       "test": {"s": [], "p": [], "v": [], "prov": []}}
game_assign = {}          # (gen, game_id) -> "train"/"test"
per_gen = {}

_real_seed(SEED)          # same starting state as split() had
train_percent, test_percent = TRP, TEP
for file_path in files:
    with h5.File(file_path, mode="r") as f:
        gen = Path(file_path).parent.name
        num_datasets = (len(f.keys()) - 1) // 3
        num_games = int(f["game_stats"][2])
        augmentations = num_datasets // num_games
        shuffled = np.random.permutation(num_games)
        n_test = int(test_percent * num_games)
        test_ids = set(shuffled[:n_test].tolist())
        remaining = shuffled[n_test:]
        train_ids = set(remaining[:int(train_percent * len(remaining))].tolist())
        per_gen[gen] = dict(num_games=num_games, aug=augmentations,
                            train_pct=train_percent, test_pct=test_percent,
                            n_train=len(train_ids), n_test=len(test_ids),
                            overlap=len(train_ids & test_ids))
        for game_id in range(num_games):
            is_test = game_id in test_ids
            if not is_test and game_id not in train_ids:
                continue
            side = "test" if is_test else "train"
            game_assign[(gen, game_id)] = side
            first = game_id * augmentations
            for ds in range(first, first + augmentations):
                b = f[f"boards_{ds}"][:]
                p = f[f"policies_{ds}"][:]
                v = f[f"values_{ds}"][:]
                assert len(b) == len(p) == len(v), (gen, ds, len(b), len(p), len(v))
                raw[side]["s"] += list(b)
                raw[side]["p"] += list(p)
                raw[side]["v"] += list(v)
                raw[side]["prov"] += [(gen, ds, r, game_id) for r in range(len(b))]
    train_percent = max(train_percent - TRD, 0.0)
    test_percent = max(test_percent - TED, 0.0)

print("\n--- item 4: recency ramp / split by game ---")
for gen, d in per_gen.items():
    print("  gen %s: games=%d aug=%d  train_pct=%.2f -> %d games (%.3f of non-test)  "
          "test_pct=%.2f -> %d games (%.3f)  train/test id overlap=%d" %
          (gen, d["num_games"], d["aug"], d["train_pct"], d["n_train"],
           d["n_train"] / max(1, d["num_games"] - d["n_test"]),
           d["test_pct"], d["n_test"], d["n_test"] / d["num_games"], d["overlap"]))

# ------------------------------------- validate the replay against the real output
rs, rp, rv, rc = DL.dedup_average(raw["train"]["s"], raw["train"]["p"], raw["train"]["v"])
es, ep, ev, ec = DL.dedup_average(raw["test"]["s"], raw["test"]["p"], raw["test"]["v"])
ok = (len(rs) == len(tr_s) and len(es) == len(te_s)
      and np.array_equal(np.asarray(rs), np.asarray(tr_s))
      and np.allclose(np.asarray(rp), np.asarray(tr_p))
      and np.allclose(np.asarray(rv), np.asarray(tr_v)))
print("\nreplay reproduces the real split byte-for-byte:", ok)
assert ok, "replay diverged - provenance below would be meaningless"

raw_train_n = len(raw["train"]["s"])
raw_test_n = len(raw["test"]["s"])
print("raw train samples %d -> %d unique (%.1f%% removed)" %
      (raw_train_n, len(rs), 100 * (1 - len(rs) / raw_train_n)))
print("raw test  samples %d -> %d unique" % (raw_test_n, len(es)))
print("weight normalisation check: counts*len/sum == returned weights:",
      np.allclose(rc * (len(rc) / rc.sum()), tr_w),
      " mean weight %.10f" % tr_w.mean())

# --------------------------------------------- item 3: trace 5 samples to raw h5
prov = raw["train"]["prov"]
flat = np.asarray(raw["train"]["s"]).reshape(raw_train_n, -1)
# map each raw row -> unique index
_, inverse = np.unique(flat, axis=0, return_inverse=True)
inverse = inverse.reshape(-1)

rng = np.random.RandomState(7)
picks = rng.choice(len(rs), 5, replace=False)
print("\n--- item 3: full traceback of 5 surviving samples ---")
for u in picks:
    members = np.where(inverse == u)[0]
    src = [prov[m] for m in members]
    # re-read the ORIGINAL bytes straight out of the h5 files, one open per gen
    h5_states, h5_pols, h5_vals = [], [], []
    by_gen = {}
    for i, (gen, ds, row, gid) in enumerate(src):
        by_gen.setdefault(gen, []).append((i, ds, row))
    slots = [None] * len(src)
    for gen, items in by_gen.items():
        try:
            with h5.File(os.path.join(ARCHIVE, gen, "Self_Play_Data.h5"), "r") as f:
                for (i, ds, row) in items:
                    slots[i] = (f[f"boards_{ds}"][row], f[f"policies_{ds}"][row],
                                f[f"values_{ds}"][row])
        except OSError as e:
            print("   OSError reading gen", gen, e)
    h5_states = np.asarray([s[0] for s in slots])
    h5_pols = [s[1] for s in slots]
    h5_vals = [s[2] for s in slots]
    same_state = all(np.array_equal(h5_states[0], s) for s in h5_states)
    matches_dedup = np.array_equal(h5_states[0], np.asarray(rs[u]))
    mp = np.asarray(h5_pols, np.float64).mean(0)
    mv = np.asarray(h5_vals, np.float64).mean(0)
    print("  uniq #%d: %d occurrences, weight_raw=%d weight_norm=%.4f" %
          (u, len(members), rc[u], tr_w[u]))
    print("     sources (gen,dataset,row,game): %s%s" %
          (src[:4], " ..." if len(src) > 4 else ""))
    print("     all source states byte-identical: %s ; equals deduped state: %s" %
          (same_state, matches_dedup))
    print("     policy: max|mean(raw h5) - deduped| = %.3e   sum=%.6f" %
          (np.abs(mp - np.asarray(rp[u])).max(), np.asarray(rp[u]).sum()))
    print("     value : mean(raw h5)=%.6f  deduped=%.6f  diff=%.3e  (raw spread %.4f)"
          % (mv[0], float(np.asarray(rv[u])[0]),
             abs(mv[0] - float(np.asarray(rv[u])[0])),
             float(np.asarray(h5_vals, np.float64).std())))
    print("     count == number of raw occurrences: %s" % (rc[u] == len(members)))

# ------------------------- item 3b: state/target ALIGNMENT invariants on raw h5
print("\n--- item 3b: state<->policy<->value alignment invariants (raw h5) ---")
bad_ply = bad_player = bad_illegal = bad_sum = bad_alt = 0
checked_rows = 0
checked_games = 0
for gen in sorted(per_gen.keys(), key=int):
    with h5.File(os.path.join(ARCHIVE, gen, "Self_Play_Data.h5"), "r") as f:
        nd = (len(f.keys()) - 1) // 3
        for ds in range(0, min(nd, 120)):
            b = f[f"boards_{ds}"][:]; p = f[f"policies_{ds}"][:]; v = f[f"values_{ds}"][:]
            checked_games += 1
            board = b[:, :, :, 3]           # channel 3 = current board
            plane = b[:, 0, 0, 0]           # channel 0 = player who just moved
            n_discs = (board != 0).sum((1, 2))
            for t in range(len(b)):
                checked_rows += 1
                if n_discs[t] != t:
                    bad_ply += 1
                if plane[t] != (1 if t % 2 == 0 else -1):
                    bad_player += 1
                full = (board[t] != 0).sum(0) == 6      # column full -> illegal
                if (p[t][full] > 1e-8).any():
                    bad_illegal += 1
                if abs(p[t].sum() - 1.0) > 1e-4:
                    bad_sum += 1
            s = np.sign(v[:, 0])
            nz = s[s != 0]
            if len(nz) > 1 and not np.all(nz[1:] == -nz[:-1]):
                bad_alt += 1
print("  rows checked %d over %d datasets" % (checked_rows, checked_games))
print("  disc-count != ply index      : %d  (off-by-one between board and row)" % bad_ply)
print("  player plane wrong for ply    : %d" % bad_player)
print("  policy mass on a FULL column  : %d" % bad_illegal)
print("  policy does not sum to 1      : %d" % bad_sum)
print("  value sign not alternating    : %d datasets" % bad_alt)
