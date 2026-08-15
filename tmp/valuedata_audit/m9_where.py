"""M9: HOW the net misses, not just by how much.

m3 says the net sits 20-200x above the floor in the endgame, where z is nearly
deterministic.  Two candidate mechanisms:
  (i)  the head is under-confident (|pred| too small where |z|=1)
  (ii) the head is confidently WRONG (sign errors)
Also checks whether dedup+count-weighting distorts the ply mass of the objective
(the Dataloader comment claims it does not).
"""
import numpy as np
import onnxruntime as ort
import common as C

QW = {"run2": 0.2, "run3": 0.4}


def run(run_name, gen):
    games = C.load_games(run_name, gen)
    if not games:
        return
    keys, z, t, ply, gid = C.stack_samples(games)
    qw = QW[run_name]
    q = (t - (1 - qw) * z) / qw
    boards = np.concatenate([g["boards"] for g in games]).astype(np.float32)
    s = ort.InferenceSession(f"{C.RUNS[run_name]}\\{gen}\\model.onnx",
                             providers=["CPUExecutionProvider"])
    iname = s.get_inputs()[0].name
    vi = [o.name for o in s.get_outputs()].index("value")
    v = np.concatenate([s.run(None, {iname: boards[i:i + 4096]})[vi].reshape(-1)
                        for i in range(0, len(boards), 4096)]).astype(np.float64)

    inv, counts, U, _ = C.group_index(keys)
    print(f"\n=== {run_name} model {gen} on gen {gen} ===")
    print(f"{'bucket':>7} {'N':>6} {'mass%':>6} | {'E|z|':>5} {'E|pred|':>7} {'E|q|':>5} | "
          f"{'sign_ok(net)':>12} {'sign_ok(q)':>10} | {'corr(net,z)':>11} {'corr(q,z)':>9}")
    for bn, lo, hi in C.BUCKETS + [("ALL", 0, 41)]:
        m = (ply >= lo) & (ply <= hi)
        d = z[m] != 0
        so_n = float((np.sign(v[m][d]) == np.sign(z[m][d])).mean())
        so_q = float((np.sign(q[m][d]) == np.sign(z[m][d])).mean())
        cn = float(np.corrcoef(v[m], z[m])[0, 1])
        cq = float(np.corrcoef(q[m], z[m])[0, 1])
        print(f"{bn:>7} {m.sum():>6} {100*m.mean():>6.1f} | {np.abs(z[m]).mean():>5.3f} "
              f"{np.abs(v[m]).mean():>7.3f} {np.abs(q[m]).mean():>5.3f} | "
              f"{so_n:>12.3f} {so_q:>10.3f} | {cn:>11.3f} {cq:>9.3f}")

    # dedup mass check: sum of counts per bucket == raw sample count per bucket?
    ply_j = np.bincount(inv, weights=ply, minlength=len(counts)) / counts
    print("  dedup+count-weight ply mass check (should equal the raw mass exactly):")
    for bn, lo, hi in C.BUCKETS:
        sel = (ply_j >= lo) & (ply_j <= hi)
        print(f"    {bn:>6}: raw {(100*((ply>=lo)&(ply<=hi)).mean()):>5.1f}%  "
              f"dedup-weighted {(100*counts[sel].sum()/len(ply)):>5.1f}%  "
              f"unique-position share {(100*sel.mean()):>5.1f}%")


if __name__ == "__main__":
    for rn, gens in (("run2", [10, 15]), ("run3", [10, 15])):
        for g in gens:
            run(rn, g)
