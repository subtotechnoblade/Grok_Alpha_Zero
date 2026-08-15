"""M3: what the SHIPPED net achieves vs the floor, on data held out by construction.

pipeline.Run: folder N holds the model that PLAYED generation N's self play, and
it was trained on the window ending at gen N-1.  So gen N's h5 is held out for
model N.  (verified by reading pipeline.Run / Train_NN: Train_NN(generation=G)
reads gens <=G and writes weights into folder G+1.)

Per-sample (un-deduped) weighted MSE, decomposed by ply bucket x occurrence-count
bucket, against BOTH the stored target and pure z, each next to:
  floor(ply model)  = E[Var(z|s)] from the exact-ply repeat estimator
  q-bound           = E[(z-q)^2], a certified upper bound on the floor
  const-0           = E[z^2]
"""
import json
import numpy as np
import onnxruntime as ort
import common as C

QW = {"run2": 0.2, "run3": 0.4}
CB = [("n=1", 1, 1), ("n=2-9", 2, 9), ("n>=10", 10, 10 ** 9)]


def sigma_curve(z, ply, inv, ng):
    """per-exact-ply pooled within-position variance of z (n>=2 groups)."""
    curve = np.full(42, np.nan)
    for p in range(42):
        m = ply == p
        if m.sum() < 2:
            continue
        si, sv = inv[m], z[m]
        n = np.bincount(si, minlength=ng).astype(np.float64)
        s = np.bincount(si, weights=sv, minlength=ng)
        s2 = np.bincount(si, weights=sv ** 2, minlength=ng)
        k = n >= 2
        df = (n[k] - 1).sum()
        if df > 0:
            curve[p] = (s2[k] - s[k] ** 2 / n[k]).sum() / df
    # plies with no repeats at all: carry the last known value forward (late game,
    # where sigma^2 is already ~0, so the choice barely matters)
    last = np.nan
    for p in range(42):
        if np.isnan(curve[p]):
            curve[p] = last
        else:
            last = curve[p]
    return curve


def eval_gen(run, gen):
    games = C.load_games(run, gen)
    if not games:
        return None
    keys, z, t, ply, gid = C.stack_samples(games)
    qw = QW[run]
    q = (t - (1.0 - qw) * z) / qw
    inv, counts, U, _ = C.group_index(keys)
    ng = len(counts)
    npos = counts[inv]
    curve = sigma_curve(z, ply, inv, ng)
    floor_s = curve[ply]

    boards = np.concatenate([g["boards"] for g in games]).astype(np.float32)
    sess = ort.InferenceSession(f"{C.RUNS[run]}\\{gen}\\model.onnx",
                                providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    onames = [o.name for o in sess.get_outputs()]
    vidx = onames.index("value")
    preds = []
    for i in range(0, len(boards), 4096):
        preds.append(sess.run(None, {iname: boards[i:i + 4096]})[vidx].reshape(-1))
    v = np.concatenate(preds).astype(np.float64)

    print(f"\n===== {run} model {gen} on gen {gen} data (held out) ; N={len(z)} "
          f"unique={U} ; pred mean={v.mean():+.4f} std={v.std():.4f} "
          f"min={v.min():+.3f} max={v.max():+.3f} =====")
    print(f"{'bucket':>7} {'count':>7} {'N':>6} | {'MSE(net,t)':>10} {'MSE(net,z)':>10} "
          f"| {'FLOOR':>7} {'q-bound':>7} {'const0':>7} {'MSE(t,z)':>8} | {'net-floor':>9} {'ratio':>6}")
    rows = []
    for bname, lo, hi in C.BUCKETS + [("ALL", 0, 41)]:
        for cname, clo, chi in CB + [("any", 1, 10 ** 9)]:
            m = (ply >= lo) & (ply <= hi) & (npos >= clo) & (npos <= chi)
            if m.sum() == 0:
                continue
            mt = float(((v[m] - t[m]) ** 2).mean())
            mz = float(((v[m] - z[m]) ** 2).mean())
            fl = float(np.nanmean(floor_s[m]))
            qb = float(((z[m] - q[m]) ** 2).mean())
            c0 = float((z[m] ** 2).mean())
            rows.append(dict(run=run, gen=gen, bucket=bname, cb=cname, N=int(m.sum()),
                             mse_net_t=mt, mse_net_z=mz, floor=fl, qbound=qb, const0=c0,
                             mse_t_z=float(((t[m] - z[m]) ** 2).mean())))
            print(f"{bname:>7} {cname:>7} {m.sum():>6} | {mt:>10.4f} {mz:>10.4f} "
                  f"| {fl:>7.4f} {qb:>7.4f} {c0:>7.4f} {rows[-1]['mse_t_z']:>8.4f} "
                  f"| {mz-fl:>9.4f} {mz/fl if fl>1e-9 else float('nan'):>6.2f}")
    return rows


if __name__ == "__main__":
    allr = []
    for run, gens in (("run2", [0, 5, 10, 15]), ("run3", [0, 5, 10, 15])):
        for g in gens:
            r = eval_gen(run, g)
            if r:
                allr += r
    with open("m3_net.json", "w") as f:
        json.dump(allr, f, indent=1)
