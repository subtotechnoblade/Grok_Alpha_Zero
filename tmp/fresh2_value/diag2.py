"""Follow-ups: (1) once-only vs repeated MSE per gen (memorization vs difficulty),
(2) net prediction on the heavy openers vs pooled/per-gen targets,
(3) noise-floor estimate for the val loss as keras computes it (deduped + count
weights), using repeated-position variance as the per-ply noise proxy.
Read-only."""
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

GENS = [8, 9, 10, 11]
NET = "Connect4/Grok_Zero_Train/12/model.onnx"
PLY_BUCKETS = [(0, 5), (6, 15), (16, 41)]


def load_gen(gen):
    path = f"Connect4/Grok_Zero_Train/{gen}/Self_Play_Data.h5"
    out = []
    try:
        with h5.File(path, "r") as f:
            n = int(f["game_stats"][2])
            aug = ((len(f.keys()) - 1) // 3) // n
            for gid in range(n):
                did = gid * aug
                out.append((f[f"boards_{did}"][:], f[f"values_{did}"][:]))
    except OSError as e:
        print(f"gen {gen} busy ({e})")
    return out


def main():
    sess = rt.InferenceSession(NET, providers=["CPUExecutionProvider"])

    def predict(boards):
        preds = []
        for i in range(0, len(boards), 1024):
            _, val = sess.run(None, {"inputs": boards[i:i + 1024].astype(np.float32)})
            preds.append(val[:, 0])
        return np.concatenate(preds)

    data = {}
    pooled = defaultdict(list)          # key -> list of (gen, value)
    ply_of = {}
    for g in GENS:
        B, V, P, K = [], [], [], []
        for boards, values in load_gen(g):
            for ply, (b, v) in enumerate(zip(boards, values)):
                B.append(b); V.append(float(v[0])); P.append(ply); K.append(b.tobytes())
                pooled[b.tobytes()].append((g, float(v[0])))
                ply_of.setdefault(b.tobytes(), ply)
        data[g] = (np.asarray(B), np.asarray(V), np.asarray(P), K)

    n_window = {k: len(v) for k, v in pooled.items()}

    # ---- (1) once-only (in the whole window) vs repeated: MSE per gen/bucket ----
    print("(1) gen-12 net MSE split by occurrence in the WHOLE 8-11 window")
    print("    once-only = position seen exactly once in all 4 gens (target = 1 noisy sample)")
    print(f"{'bucket':>8} {'class':>10}" + "".join(f"{f'gen {g}':>12}" for g in GENS))
    for lo, hi in PLY_BUCKETS:
        for cls, cond in [("once", lambda n: n == 1), ("repeated", lambda n: n > 1)]:
            row = f"{f'{lo}-{hi}':>8} {cls:>10}"
            for g in GENS:
                B, V, P, K = data[g]
                pr = getattr(main, f"_pr{g}", None)
                if pr is None:
                    pr = predict(B)
                    setattr(main, f"_pr{g}", pr)
                m = (P >= lo) & (P <= hi) & np.array([cond(n_window[k]) for k in K])
                row += (f"{np.mean((pr[m] - V[m]) ** 2):>9.4f}({m.sum():>4})"
                        if m.any() else f"{'-':>12}")
            print(row)

    # ---- (2) heavy openers: net prediction vs pooled and per-gen targets ----
    print("\n(2) net prediction on the heaviest positions (count in window, un-mirrored)")
    heavy = sorted(pooled.items(), key=lambda kv: -len(kv[1]))[:8]
    print(f"{'ply':>4} {'n':>5} {'net pred':>9} {'pooled tgt':>11}"
          + "".join(f"{f'g{g} tgt':>9}" for g in GENS))
    boards_by_key = {}
    for g in GENS:
        B, V, P, K = data[g]
        for i, k in enumerate(K):
            if k not in boards_by_key:
                boards_by_key[k] = B[i]
    hb = np.asarray([boards_by_key[k] for k, _ in heavy])
    hp = predict(hb)
    for (k, occ), p in zip(heavy, hp):
        vals = np.array([v for _, v in occ])
        gcols = ""
        for g in GENS:
            gv = [v for gg, v in occ if gg == g]
            gcols += f"{np.mean(gv):>+9.3f}" if gv else f"{'-':>9}"
        print(f"{ply_of[k]:>4} {len(occ):>5} {p:>+9.3f} {vals.mean():>+11.3f}{gcols}")

    # ---- (3) noise floor of the val loss as computed (dedup + count weights) ----
    # keras val value loss = sum_i w_i (pred_i - t_i)^2 / sum_i w_i, w_i = n_i (mean-1
    # normalized), t_i = mean of n_i draws. Perfect prediction of the pooled-truth
    # leaves E[(t_i - mu_i)^2] = sigma_i^2 / n_i, so floor = sum_i n_i * sigma_i^2/n_i
    # / sum_i n_i = mean per-sample sigma^2. Estimate sigma^2 per bucket from
    # POOLED-window positions with n>=2 (unbiased, includes between-gen drift) and
    # from WITHIN-single-gen repeats (drift-free).
    print("\n(3) per-sample outcome variance (sigma^2) estimates by ply bucket")
    print(f"{'bucket':>8} {'pooled n>=2':>12} {'within-gen n>=2':>16} "
          f"{'samples in bucket':>18}")
    tot = sum(len(v) for v in pooled.values())
    floor_pooled = floor_within = 0.0
    for lo, hi in PLY_BUCKETS:
        pv, wv = [], []
        nb = 0
        for k, occ in pooled.items():
            if not (lo <= ply_of[k] <= hi):
                continue
            nb += len(occ)
            vals = np.array([v for _, v in occ])
            if len(vals) >= 2:
                pv.append(vals.var(ddof=1))
            bygen = defaultdict(list)
            for gg, v in occ:
                bygen[gg].append(v)
            for gvals in bygen.values():
                if len(gvals) >= 2:
                    wv.append(np.var(gvals, ddof=1))
        pv_m = np.mean(pv) if pv else float("nan")
        wv_m = np.mean(wv) if wv else float("nan")
        print(f"{f'{lo}-{hi}':>8} {pv_m:>12.4f} {wv_m:>16.4f} {nb:>18}")
        floor_pooled += pv_m * nb / tot
        floor_within += wv_m * nb / tot
    print(f"\nimplied FLOOR of the val value loss (count-weighted dedup targets):")
    print(f"  using pooled-window sigma^2 (drift included) : {floor_pooled:.4f}")
    print(f"  using within-gen sigma^2 (drift removed)     : {floor_within:.4f}")
    print(f"  observed best val value loss                  : ~0.247")
    print("  caveat: sigma^2 proxied from repeated positions; once-only positions")
    print("  (76% of uniques) may differ - repeated ones are opening-heavy.")


if __name__ == "__main__":
    main()
