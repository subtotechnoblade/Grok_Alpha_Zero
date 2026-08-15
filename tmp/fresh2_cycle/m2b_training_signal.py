"""M2b: the EXACT value target the net was trained on for each 1-stone state.

Key structural facts (verified in code):
- augment_sample mirrors every game; values are copied unmirrored. So the
  1-stone-at-c state receives targets from games opened at c AND at 6-c.
- The ply-1 input state is IDENTICAL across all games opening at c (history
  planes are zero at ply 1), so dedup_average collapses it to ONE sample whose
  target is the occurrence-weighted mean. => training target for col c and
  col 6-c are EXACTLY equal. Any net asymmetry between mirror twins is fit error.
- Window: gens g-1..g-4, kept fraction 1.0/0.9/0.8/0.7 (subtractive decay).

D(g,c) = decay-weighted pooled mean of -values_[1] (opener persp) over the
window, pooling openings at c and 6-c. Compare to net-g leaf opener-Q.
"""
import os, sys
ROOT = r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np
import h5py as h5

# per-gen per-column: sum of v1 (p+1 persp) and counts, unaugmented games
sums = {}; cnts = {}
for gen in range(0, 13):
    fp = f"Connect4/Grok_Zero_Train/{gen}/Self_Play_Data.h5"
    if not os.path.exists(fp):
        continue
    try:
        with h5.File(fp, "r") as f:
            n = int(f["game_stats"][2])
            n_sets = (len(f.keys()) - 1) // 3
            if n <= 0 or n_sets % n:
                continue
            aug = n_sets // n
            s = np.zeros(7); c = np.zeros(7)
            for gid in range(n):
                did = gid * aug
                b1 = f[f"boards_{did}"][1][:, :, 3]
                col = int(np.argwhere(b1 != 0)[0][1])
                s[col] += float(f[f"values_{did}"][1][0])
                c[col] += 1
        sums[gen] = s; cnts[gen] = c
    except OSError:
        continue

qtab = np.load(r"tmp\fresh2_cycle\opener_q.npy")  # rows: gen, q0..q6, p0..p6
qs = {int(r[0]): r[1:8] for r in qtab}

print("D(g,c): dedup training target (opener persp) for the 1-stone state, vs net leaf opener-Q")
for g in range(6, 13):
    ws = [1.0, 0.9, 0.8, 0.7]
    num = np.zeros(7); den = np.zeros(7)
    for k, w in zip(range(1, 5), ws):
        src = g - k
        if src not in sums:
            continue
        num += w * sums[src]
        den += w * cnts[src]
    # mirror pooling: state at col c trained by openings at c and 6-c
    D = np.full(7, np.nan)
    for c in range(7):
        nn = num[c] + num[6 - c] if c != 3 else num[3]
        dd = den[c] + den[6 - c] if c != 3 else den[3]
        if dd > 0:
            D[c] = -nn / dd   # flip p+1 -> opener perspective
    print(f"\n--- gen {g} net (window gens {max(g-4,0)}..{g-1}) ---")
    print("            " + "".join(f"  col{c}  " for c in range(7)))
    print("train tgt D " + "".join(f" {D[c]:6.3f} " for c in range(7)))
    if g in qs:
        q = qs[g]
        print("net leaf Q  " + "".join(f" {q[c]:6.3f} " for c in range(7)))
        m = ~np.isnan(D)
        r = np.corrcoef(D[m], q[m])[0, 1]
        # mirror-twin fit error: targets identical, so |q(c)-q(6-c)| is pure error
        twin = [abs(q[c] - q[6 - c]) for c in (0, 1, 2)]
        print(f"corr(D, netQ) = {r:+.3f}   mirror-twin |dQ| fit error: "
              f"0/6={twin[0]:.3f} 1/5={twin[1]:.3f} 2/4={twin[2]:.3f}")

# per-gen outcome table for the report (pooled mirror classes, single gen)
print("\nsingle-gen pooled outcome (opener persp, mirror classes), games in ():")
for g in sorted(sums):
    s, c = sums[g], cnts[g]
    row = []
    for cls in ((0, 6), (1, 5), (2, 4), (3,)):
        cc = sum(c[i] for i in cls)
        vv = -sum(s[i] for i in cls) / cc if cc else np.nan
        row.append(f"{'/'.join(map(str,cls))}: {vv:+.3f} ({int(cc)})")
    print(f"gen {g:>2}  " + "   ".join(row))
