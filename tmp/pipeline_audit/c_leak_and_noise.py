"""ITEM 3 (high-count traceback) + ITEM 4 (leakage) + label-noise floor + ITEM 5.

Re-runs the deterministic replay from b_dataloader.py, then:
  * traces the HEAVY dedup groups (the ones the weighting actually acts on)
  * measures how much of the TEST set is byte-identical (and mirror-identical)
    to something in TRAIN  -> whether val_loss can detect overfitting at all
  * measures the irreducible val MSE floor from label disagreement
  * checks the __getitem__ contract
"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
REPO = r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero"
sys.path.insert(0, REPO)

import numpy as np, h5py as h5
from pathlib import Path

SEED = 12345
_real_seed = np.random.seed
np.random.seed = lambda *a, **k: _real_seed(SEED)
import Dataloader as DL

ARCHIVE = os.path.join(REPO, "tmp", "run3_archive_gen0-15_bottleneck_preweightfix")
NPG, TRP, TRD, TEP, TED = 4, 1.0, 0.2, 0.1, 0.02

splitter = DL.Create_Train_Test_Split(ARCHIVE, NPG, TRP, TRD, TEP, TED, dedup_average=False)
files = splitter.files

raw = {"train": {"s": [], "p": [], "v": [], "prov": []},
       "test": {"s": [], "p": [], "v": [], "prov": []}}
_real_seed(SEED)
train_percent, test_percent = TRP, TEP
for file_path in files:
    with h5.File(file_path, mode="r") as f:
        gen = Path(file_path).parent.name
        nd = (len(f.keys()) - 1) // 3
        ng = int(f["game_stats"][2]); aug = nd // ng
        shuffled = np.random.permutation(ng)
        n_test = int(test_percent * ng)
        test_ids = set(shuffled[:n_test].tolist())
        rem = shuffled[n_test:]
        train_ids = set(rem[:int(train_percent * len(rem))].tolist())
        for gid in range(ng):
            is_test = gid in test_ids
            if not is_test and gid not in train_ids:
                continue
            side = "test" if is_test else "train"
            for ds in range(gid * aug, gid * aug + aug):
                b = f[f"boards_{ds}"][:]; p = f[f"policies_{ds}"][:]; v = f[f"values_{ds}"][:]
                raw[side]["s"] += list(b); raw[side]["p"] += list(p); raw[side]["v"] += list(v)
                raw[side]["prov"] += [(gen, ds, r, gid) for r in range(len(b))]
    train_percent = max(train_percent - TRD, 0.0)
    test_percent = max(test_percent - TED, 0.0)

trs, trp, trv, trc = DL.dedup_average(raw["train"]["s"], raw["train"]["p"], raw["train"]["v"])
tes, tep_, tev, tec = DL.dedup_average(raw["test"]["s"], raw["test"]["p"], raw["test"]["v"])
trw = trc * (len(trc) / trc.sum())
tew = tec * (len(tec) / tec.sum())
trs_a, trp_a, trv_a = np.asarray(trs), np.asarray(trp), np.asarray(trv)
tes_a, tep_a, tev_a = np.asarray(tes), np.asarray(tep_), np.asarray(tev)
print("train uniq %d  test uniq %d" % (len(trs_a), len(tes_a)))

# ---------------------------- heavy-group traceback (what the weights act on)
flat = np.asarray(raw["train"]["s"]).reshape(len(raw["train"]["s"]), -1)
_, inverse = np.unique(flat, axis=0, return_inverse=True)
inverse = inverse.reshape(-1)
prov = raw["train"]["prov"]
order = np.argsort(-trc)
picks = list(order[:2]) + list(np.random.RandomState(3).choice(
    np.where((trc >= 4) & (trc <= 40))[0], 3, replace=False))
print("\n--- item 3: traceback of HEAVY dedup groups ---")
for u in picks:
    members = np.where(inverse == u)[0]
    src = [prov[m] for m in members]
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
            print("  OSError", gen, e)
    S = np.asarray([s[0] for s in slots]); P = np.asarray([s[1] for s in slots], np.float64)
    V = np.asarray([s[2] for s in slots], np.float64)
    ply = int((S[0][:, :, 3] != 0).sum())
    gens = {}
    for (g, d, r, gid) in src:
        gens[g] = gens.get(g, 0) + 1
    print("  uniq #%d  ply=%d  count=%d  weight=%.2f   per-gen occurrences %s" %
          (u, ply, trc[u], trw[u], dict(sorted(gens.items()))))
    print("     every source state identical to deduped state: %s" %
          bool(np.all(S == np.asarray(trs[u]))))
    print("     policy  max|mean(h5) - deduped| = %.3e" %
          np.abs(P.mean(0) - trp_a[u]).max())
    print("     value   mean(h5) = %+.6f   deduped = %+.6f   |diff| = %.3e" %
          (V.mean(), trv_a[u][0], abs(V.mean() - trv_a[u][0])))
    print("     raw value spread in the group: std %.4f  min %+.3f max %+.3f" %
          (V.std(), V.min(), V.max()))
    print("     count == occurrences: %s   weight == count*len/sum: %s" %
          (trc[u] == len(members), np.isclose(trw[u], trc[u] * len(trc) / trc.sum())))

# --------------------------------------------- leakage: test states seen in train
tr_keys = {trs_a[i].tobytes(): i for i in range(len(trs_a))}
hit = np.array([tr_keys.get(tes_a[i].tobytes(), -1) for i in range(len(tes_a))])
mask = hit >= 0
print("\n--- item 4: does the val split contain positions the net trained on? ---")
print("  test unique states also present VERBATIM in train: %d / %d = %.1f%%" %
      (mask.sum(), len(mask), 100 * mask.mean()))
print("  share of the TEST LOSS MASS (sum of weights) on those: %.1f%%" %
      (100 * tew[mask].sum() / tew.sum()))
print("  share of raw (pre-dedup) test samples on those:        %.1f%%" %
      (100 * tec[mask].sum() / tec.sum()))
mir = tes_a[:, :, ::-1, :]
hit_m = np.array([tr_keys.get(mir[i].copy().tobytes(), -1) for i in range(len(tes_a))])
both = mask | (hit_m >= 0)
print("  including left-right MIRRORS of train states:          %.1f%% of unique, "
      "%.1f%% of test loss mass" % (100 * both.mean(), 100 * tew[both].sum() / tew.sum()))

# ------------------------------------------------ irreducible val-MSE floor
i_te = np.where(mask)[0]; i_tr = hit[mask]
dv = tev_a[i_te, 0] - trv_a[i_tr, 0]
w = tew[i_te]
print("\n--- label-noise floor on the SHARED positions ---")
print("  E[(v_test - v_train)^2] weighted = %.4f   unweighted = %.4f" %
      ((w * dv ** 2).sum() / w.sum(), (dv ** 2).mean()))
print("  test value target variance (weighted)          = %.4f" %
      ((tew * (tev_a[:, 0] - (tew * tev_a[:, 0]).sum() / tew.sum()) ** 2).sum() / tew.sum()))
print("  a perfect memoriser of the TRAIN targets scores val MSE >= %.4f on the "
      "shared part alone" % ((w * dv ** 2).sum() / w.sum()))
print("  value target stats: train mean %+.4f std %.4f | test mean %+.4f std %.4f" %
      (trv_a[:, 0].mean(), trv_a[:, 0].std(), tev_a[:, 0].mean(), tev_a[:, 0].std()))
print("  fraction of |v| == 1.0 exactly: train %.4f  (q_weight==0 would make this ~1)"
      % (np.mean(np.abs(np.asarray(raw['train']['v'])[:, 0]) == 1.0)))
print("  raw (pre-dedup) |v| histogram edges:",
      np.round(np.percentile(np.abs(np.asarray(raw['train']['v'])[:, 0]),
                             [0, 25, 50, 75, 100]), 4))

# ------------------------------------------------------- item 5: __getitem__
print("\n--- item 5: __getitem__ contract ---")
BS = 1024
dl = DL.Dataloader(trs, trp, trv, BS, weights=trw)
print("  num_samples %d  batch_size %d  len(dl) %d  expected %d" %
      (dl.num_samples, BS, len(dl), int(np.ceil(dl.num_samples / BS))))
sizes = [len(dl.batch_indexes[i]) for i in range(len(dl))]
print("  batch sizes: first %d, last %d, sum %d == num_samples %s" %
      (sizes[0], sizes[-1], sum(sizes), sum(sizes) == dl.num_samples))
allidx = np.concatenate([dl.batch_indexes[i] for i in range(len(dl))])
print("  every sample used exactly once per epoch: %s" %
      (len(np.unique(allidx)) == dl.num_samples))
x, (p, v), (w1, w2) = dl[0]
print("  shapes  x %s %s | p %s %s | v %s %s | w %s %s" %
      (x.shape, x.dtype, p.shape, p.dtype, v.shape, v.dtype, w1.shape, w1.dtype))
print("  weights tuple identical for both heads: %s" % np.array_equal(w1, w2))
idx = dl.batch_indexes[0]
ok_x = np.array_equal(x, np.asarray(trs)[idx])
ok_p = np.array_equal(p, np.asarray(trp)[idx])
ok_v = np.array_equal(v, np.asarray(trv)[idx])
ok_w = np.array_equal(w1, trw[idx])
print("  x/p/v/w all drawn with the SAME index vector: %s %s %s %s" %
      (ok_x, ok_p, ok_v, ok_w))
before = dl.batch_indexes[0].copy()
dl.on_epoch_end()
print("  on_epoch_end reshuffles: %s ; still a permutation: %s" %
      (not np.array_equal(before, dl.batch_indexes[0]),
       len(np.unique(np.concatenate([dl.batch_indexes[i] for i in range(len(dl))])))
       == dl.num_samples))
# tiny dataset / partial-batch edge cases
for n, bs in [(5, 1024), (1024, 1024), (1025, 1024), (2047, 1024)]:
    d = DL.Dataloader(trs[:n], trp[:n], trv[:n], bs, weights=trw[:n])
    s = [len(d.batch_indexes[i]) for i in range(len(d))]
    print("  n=%-5d bs=%d -> %d batches %s  total %d" % (n, bs, len(d), s, sum(s)))
