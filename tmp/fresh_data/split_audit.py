"""Dataloader-side audit, run against the REAL Create_Train_Test_Split.

1. Window + decay: patch np.random.permutation to identity so the game selection
   is deterministic, predict per-generation game/sample counts under the code's
   SUBTRACTIVE decay (train_percent -= decay), and compare with what the real
   split() actually collects. Also compute what the config comment's
   MULTIPLICATIVE decay (pct *= decay) would have selected, for contrast.
2. Empty-board sample after dedup_average: exact left-right symmetry check.
3. Dedup decomposition on the train-side pool: mirror-self vs within-gen
   cross-game vs cross-gen duplicates.
4. Occurrence-weighted-mean synthetic test of dedup_average.
"""
import sys
import numpy as np
import h5py as h5

sys.path.insert(0, ".")

# deterministic game selection BEFORE importing/running the split
np.random.permutation = lambda n: np.arange(n)
np.random.seed = lambda *a, **k: None

from Dataloader import Create_Train_Test_Split, dedup_average
from Connect4.game import Connect4

BASE = "Connect4/Grok_Zero_Train"

# ---------- predicted selection under the code's arithmetic ----------
files = list(range(9, 5, -1))  # gen 9 window after [-5:], drop oldest, reverse -> [9,8,7,6]
game_counts = {}
for g in files:
    with h5.File(f"{BASE}/{g}/Self_Play_Data.h5", "r") as f:
        game_counts[g] = int(f["game_stats"][2])

print("=" * 70)
print("PART 1: window + decay arithmetic (code = subtractive, comment = multiplicative)")
print("=" * 70)
tp, sp = 1.0, 0.1           # train_percent, test_percent as the code evolves them
mtp, msp = 1.0, 0.1         # what the comment (multiplicative) would give
pred = {}
for g in files:
    n = game_counts[g]
    ntest = int(sp * n)
    ntrain = int(tp * (n - ntest))
    m_ntest = int(msp * n)
    m_ntrain = int(mtp * (n - m_ntest))
    pred[g] = (ntest, ntrain)
    print(f"gen {g}: games={n}  CODE(train%={tp:.2f}, test%={sp:.2f}) -> train {ntrain:4d} games, test {ntest:3d}"
          f"   COMMENT-INTENT(x{0.1}) -> train {m_ntrain:4d}, test {m_ntest:3d}")
    tp = max(tp - 0.1, 0.0); sp = max(sp - 0.01, 0.0)
    mtp = mtp * 0.1; msp = msp * 0.01

# per-gen selected ids under identity permutation
def selected_ids(gen):
    n = game_counts[gen]
    ntest, ntrain = pred[gen]
    test_ids = list(range(ntest))
    train_ids = list(range(ntest, ntest + ntrain))
    return test_ids, train_ids

# predicted sample counts
pred_train_samples = pred_test_samples = 0
per_gen_train_samples = {}
for g in files:
    test_ids, train_ids = selected_ids(g)
    with h5.File(f"{BASE}/{g}/Self_Play_Data.h5", "r") as f:
        augs = ((len(f.keys()) - 1) // 3) // game_counts[g]
        cnt_tr = sum(f[f"boards_{gid * augs}"].shape[0] for gid in train_ids) * augs
        cnt_te = sum(f[f"boards_{gid * augs}"].shape[0] for gid in test_ids) * augs
    per_gen_train_samples[g] = cnt_tr
    pred_train_samples += cnt_tr
    pred_test_samples += cnt_te
print(f"\npredicted (subtractive) train samples: {pred_train_samples}, test samples: {pred_test_samples}")

# ---------- run the REAL split ----------
split = Create_Train_Test_Split(BASE, 4, 1.0, 0.1, 0.1, 0.01, dedup_average=False)
print("real split file order:", [p.replace("\\", "/").split("/")[-2] for p in split.files])
(tr_s, tr_p, tr_v), (te_s, te_p, te_v) = split.split()
print(f"REAL split: train samples = {len(tr_s)}, test = {len(te_s)}  "
      f"(match predicted: {len(tr_s) == pred_train_samples and len(te_s) == pred_test_samples})")
frac = {g: per_gen_train_samples[g] / pred_train_samples for g in files}
print("train-side sample share by gen:", {g: f"{v:.1%}" for g, v in frac.items()})

# ---------- PART 2: dedup + empty-board symmetry ----------
print()
print("=" * 70)
print("PART 2: dedup_average on the real train side + empty-board symmetry")
print("=" * 70)
n_before = len(tr_s)
d_s, d_p, d_v = dedup_average(tr_s, tr_p, tr_v)
print(f"train {n_before} -> {len(d_s)} unique ({1 - len(d_s)/n_before:.1%} removed)")

empty_state = Connect4().get_input_state()
hits = [i for i, s in enumerate(d_s) if np.array_equal(s, empty_state)]
print(f"empty-board samples found after dedup: {len(hits)}")
for i in hits:
    p = d_p[i]
    v = d_v[i]
    sym = np.allclose(p, p[::-1], atol=0)          # exact
    sym_close = np.allclose(p, p[::-1], atol=1e-6)  # float tolerance
    with np.printoptions(precision=4, suppress=True):
        print("  policy target:", p)
        print("  flipped:      ", p[::-1])
    print(f"  exactly symmetric: {sym}   symmetric@1e-6: {sym_close}   value target: {v.ravel()[0]:+.4f}")

# ---------- PART 3: duplicate decomposition on the train-side pool ----------
print()
print("=" * 70)
print("PART 3: duplicate decomposition (train-side pool, gens 9,8,7,6)")
print("=" * 70)
# rebuild the same pool with per-sample (gen, game) identity
recs_gen = []
recs_state = []
mirror_self = {g: 0 for g in files}
for g in files:
    _, train_ids = selected_ids(g)
    with h5.File(f"{BASE}/{g}/Self_Play_Data.h5", "r") as f:
        augs = ((len(f.keys()) - 1) // 3) // game_counts[g]
        for gid in train_ids:
            b0 = f[f"boards_{gid * augs}"][:]
            b1 = f[f"boards_{gid * augs + 1}"][:]
            mirror_self[g] += int(np.sum(np.all(b0 == b1, axis=(1, 2, 3))))
            for arr in (b0, b1):
                recs_state.append(arr.reshape(arr.shape[0], -1))
                recs_gen.append(np.full(arr.shape[0], g, dtype=np.int8))
states = np.concatenate(recs_state, 0)
gens = np.concatenate(recs_gen, 0)
assert states.shape[0] == n_before, (states.shape[0], n_before)

N = states.shape[0]
U_pool = np.unique(states, axis=0).shape[0]
within = {}
for g in files:
    sg = states[gens == g]
    within[g] = (sg.shape[0], np.unique(sg, axis=0).shape[0])
sum_Ug = sum(u for _, u in within.values())
removed_total = N - U_pool
removed_within = sum(n - u for n, u in within.values())
removed_cross_gen = sum_Ug - U_pool
removed_mirror = sum(mirror_self.values())
print(f"pooled: N={N}  unique={U_pool}  removed={removed_total} ({removed_total/N:.1%})")
for g in files:
    n, u = within[g]
    print(f"  gen {g}: N={n:6d} unique={u:6d} removed-within={n-u:6d} "
          f"(of which mirror-self pairs: {mirror_self[g]})")
print(f"decomposition of {removed_total} removed duplicates:")
print(f"  (a) mirror-augmentation self-duplicates:        {removed_mirror:6d} ({removed_mirror/removed_total:.1%})")
print(f"  (b) other within-generation (cross-game/ply):   {removed_within - removed_mirror:6d} "
      f"({(removed_within - removed_mirror)/removed_total:.1%})")
print(f"  (c) cross-generation:                           {removed_cross_gen:6d} ({removed_cross_gen/removed_total:.1%})")

# ---------- PART 4: occurrence weighting synthetic ----------
print()
print("=" * 70)
print("PART 4: dedup_average occurrence weighting synthetic test")
print("=" * 70)
A = np.zeros((2, 2), dtype=np.int8); B = np.ones((2, 2), dtype=np.int8)
sts = [A, A, A, B]
pols = [np.array([1, 0], np.float32), np.array([1, 0], np.float32),
        np.array([0, 1], np.float32), np.array([0, 1], np.float32)]
vals = [np.array([1.0], np.float32), np.array([1.0], np.float32),
        np.array([0.0], np.float32), np.array([0.5], np.float32)]
ds, dp, dv = dedup_average(sts, pols, vals)
for s, p, v in zip(ds, dp, dv):
    print(f"  state[0,0]={s.ravel()[0]}  policy={p}  value={v}")
print("  expected: A (3 occurrences) -> policy [0.667, 0.333], value 0.667 (occurrence-weighted)")
print("            B (1 occurrence)  -> policy [0, 1], value 0.5")
