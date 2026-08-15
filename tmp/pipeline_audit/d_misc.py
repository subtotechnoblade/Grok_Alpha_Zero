"""Follow-ups: on_epoch_end reshuffling WITHOUT any seed patch, augmentation
board<->policy consistency, leakage on the second archive, non-shared variance."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
REPO = r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero"
sys.path.insert(0, REPO)
import numpy as np, h5py as h5
from pathlib import Path
import Dataloader as DL

# ---------------------------------------------- on_epoch_end, UNPATCHED numpy
s = [np.zeros((6, 7, 4), np.int8) + i for i in range(50)]
p = [np.zeros(7, np.float32) + i for i in range(50)]
v = [np.zeros(1, np.float32) + i for i in range(50)]
d = DL.Dataloader(s, p, v, 8)
perms = []
for _ in range(5):
    perms.append(np.concatenate([d.batch_indexes[i] for i in range(len(d))]).copy())
    d.on_epoch_end()
uniq = len({tuple(x) for x in perms})
print("on_epoch_end (unpatched RNG): %d distinct permutations out of 5" % uniq)
print("each is a valid permutation:",
      all(sorted(x.tolist()) == list(range(50)) for x in perms))
x, (pp, vv), (w1, w2) = d[0]
idx = d.batch_indexes[0]
print("x/p/v consistent after reshuffle:",
      np.array_equal(x[:, 0, 0, 0], idx.astype(np.int8)),
      np.array_equal(pp[:, 0], idx.astype(np.float32)),
      np.array_equal(vv[:, 0], idx.astype(np.float32)))

# -------------------------------------- augmentation: board and policy flipped together
print("\n--- augmentation consistency (dataset 2g vs 2g+1) ---")
bad_b = bad_p = bad_v = 0
tot = 0
for arch in ["run3_archive_gen0-15_bottleneck_preweightfix",
             "run2_archive_gen0-19_oldvaluehead"]:
    for gen in ["15", "10", "5"]:
        fp = os.path.join(REPO, "tmp", arch, gen, "Self_Play_Data.h5")
        try:
            with h5.File(fp, "r") as f:
                nd = (len(f.keys()) - 1) // 3
                for g in range(0, min(nd // 2, 100)):
                    b0, b1 = f[f"boards_{2*g}"][:], f[f"boards_{2*g+1}"][:]
                    p0, p1 = f[f"policies_{2*g}"][:], f[f"policies_{2*g+1}"][:]
                    v0, v1 = f[f"values_{2*g}"][:], f[f"values_{2*g+1}"][:]
                    tot += 1
                    if not np.array_equal(b1, b0[:, :, ::-1, :]):
                        bad_b += 1
                    if not np.allclose(p1, p0[:, ::-1]):
                        bad_p += 1
                    if not np.array_equal(v1, v0):
                        bad_v += 1
        except OSError as e:
            print("  OSError", fp, e)
print("  %d game pairs checked: board-flip mismatches %d, policy-flip mismatches %d, "
      "value-copy mismatches %d" % (tot, bad_b, bad_p, bad_v))

# --------------------------------------------------- leakage on both archives
def leak(archive, npg=4, trp=1.0, trd=0.2, tep=0.1, ted=0.02, seed=999):
    sp = DL.Create_Train_Test_Split(archive, npg, trp, trd, tep, ted, dedup_average=False)
    files = sp.files
    tr, te = {"s": [], "v": []}, {"s": [], "v": []}
    np.random.seed(seed)
    a, b = trp, tep
    for fpth in files:
        try:
            with h5.File(fpth, "r") as f:
                nd = (len(f.keys()) - 1) // 3
                ng = int(f["game_stats"][2])
                if ng <= 0 or nd % ng:
                    continue
                aug = nd // ng
                sh = np.random.permutation(ng)
                nt = int(b * ng)
                tid = set(sh[:nt].tolist()); rem = sh[nt:]
                gid_train = set(rem[:int(a * len(rem))].tolist())
                for gid in range(ng):
                    ist = gid in tid
                    if not ist and gid not in gid_train:
                        continue
                    tgt = te if ist else tr
                    for ds in range(gid * aug, gid * aug + aug):
                        tgt["s"].append(f[f"boards_{ds}"][:])
                        tgt["v"].append(f[f"values_{ds}"][:])
        except OSError as e:
            print("  OSError", fpth, e)
        a = max(a - trd, 0.0); b = max(b - ted, 0.0)
    TS = np.concatenate(tr["s"]); TV = np.concatenate(tr["v"])
    ES = np.concatenate(te["s"]); EV = np.concatenate(te["v"])
    us, ui, uc = np.unique(TS.reshape(len(TS), -1), axis=0,
                           return_inverse=True, return_counts=True)
    tv = np.zeros(len(us)); np.add.at(tv, ui.reshape(-1), TV[:, 0]); tv /= uc
    es, ei, ec = np.unique(ES.reshape(len(ES), -1), axis=0,
                           return_inverse=True, return_counts=True)
    ev = np.zeros(len(es)); np.add.at(ev, ei.reshape(-1), EV[:, 0]); ev /= ec
    keys = {us[i].tobytes(): i for i in range(len(us))}
    hit = np.array([keys.get(es[i].tobytes(), -1) for i in range(len(es))])
    m = hit >= 0
    w = ec / ec.mean()
    shared_mass = w[m].sum() / w.sum()
    dv = ev[m] - tv[hit[m]]
    floor = (w[m] * dv ** 2).sum() / w[m].sum()
    mu = (w * ev).sum() / w.sum()
    var_ns = ((w[~m] * (ev[~m] - mu) ** 2).sum() / max(w[~m].sum(), 1e-9)) if (~m).any() else 0.0
    var_all = (w * (ev - mu) ** 2).sum() / w.sum()
    print("\n%s" % Path(archive).name)
    print("  gens used: %s" % [Path(x).parent.name for x in files])
    print("  train raw %d -> %d uniq | test raw %d -> %d uniq" %
          (len(TS), len(us), len(ES), len(es)))
    print("  test uniq states also in train: %.1f%%   test LOSS MASS on them: %.1f%%" %
          (100 * m.mean(), 100 * shared_mass))
    print("  label-noise floor on shared part  E[(v_te - v_tr)^2] = %.4f" % floor)
    print("  test target variance: all %.4f | non-shared part %.4f" % (var_all, var_ns))
    print("  => val MSE of an exact train-memoriser >= %.4f  (vs %.4f for predicting "
          "the mean) : an %.0f%% 'improvement' that is pure memorisation" %
          (shared_mass * floor + (1 - shared_mass) * var_ns, var_all,
           100 * (1 - (shared_mass * floor + (1 - shared_mass) * var_ns) / var_all)))


leak(os.path.join(REPO, "tmp", "run3_archive_gen0-15_bottleneck_preweightfix"))
leak(os.path.join(REPO, "tmp", "run2_archive_gen0-19_oldvaluehead"))
