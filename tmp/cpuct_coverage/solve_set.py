"""Build a deduped set of self-play positions and solve them EXACTLY (weak/sign solve).

The set is drawn evenly across every c_puct shard so the position pool itself is not
biased toward one configuration.

usage: solve_set.py <run_dir> <out.npz> [n_per_cp] [ply_min] [ply_max] [node_cap] [workers]
"""
import sys
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp/cpuct_coverage"))

_S = None


def _solve_chunk(args):
    boards, movers, node_cap = args
    from solver import Solver
    global _S
    if _S is None:
        _S = Solver()
    out = np.full((len(boards), 8), 999, dtype=np.int64)  # 7 signs + ok
    for i in range(len(boards)):
        signs, ok = _S.classify(boards[i].reshape(6, 7), int(movers[i]), node_cap)
        out[i, :7] = signs
        out[i, 7] = int(ok)
    return out


def main():
    run = Path(sys.argv[1])
    out = sys.argv[2]
    n_per_cp = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    ply_min = int(sys.argv[4]) if len(sys.argv) > 4 else 14
    ply_max = int(sys.argv[5]) if len(sys.argv) > 5 else 34
    node_cap = int(sys.argv[6]) if len(sys.argv) > 6 else 40_000_000
    workers = int(sys.argv[7]) if len(sys.argv) > 7 else 5

    rng = np.random.default_rng(0)
    pool_b, pool_m, pool_p, pool_cp, pool_h = [], [], [], [], []
    by_cp = {}
    for p in sorted(run.glob("cp*_b*_w*.npz")):
        d = np.load(p)
        cp = float(d["meta"][0])
        by_cp.setdefault(cp, []).append(p)
    for cp, files in sorted(by_cp.items()):
        B, M, P, HH = [], [], [], []
        for p in files:
            d = np.load(p)
            ply = d["gid_ply"][:, 1]
            expl = d["ints"][:, 1]
            m = (ply >= ply_min) & (ply <= ply_max) & (expl == 0)
            B.append(d["boards"][m])
            M.append(d["movers"][m, 0])
            P.append(ply[m])
            HH.append(d["hists"][m])
        B = np.concatenate(B); M = np.concatenate(M); P = np.concatenate(P)
        HH = np.concatenate(HH)
        idx = rng.permutation(len(B))[:n_per_cp]
        pool_b.append(B[idx]); pool_m.append(M[idx]); pool_p.append(P[idx])
        pool_h.append(HH[idx])
        pool_cp.append(np.full(len(idx), cp))
        print(f"c_puct={cp}: {len(B)} eligible positions, sampled {len(idx)}", flush=True)

    boards = np.concatenate(pool_b)
    movers = np.concatenate(pool_m)
    plies = np.concatenate(pool_p)
    cps = np.concatenate(pool_cp)
    hists = np.concatenate(pool_h)

    # dedup on (board, mover) so the solver never repeats work
    keyarr = np.concatenate([boards, movers.reshape(-1, 1)], axis=1)
    _, uidx = np.unique(keyarr, axis=0, return_index=True)
    uidx = np.sort(uidx)
    print(f"{len(boards)} sampled -> {len(uidx)} unique (board, mover)", flush=True)
    boards, movers, plies, cps = boards[uidx], movers[uidx], plies[uidx], cps[uidx]
    hists = hists[uidx]

    chunks = np.array_split(np.arange(len(boards)), max(workers * 4, 1))
    args = [(boards[c], movers[c], node_cap) for c in chunks if len(c)]
    t0 = time.time()
    with mp.Pool(workers) as pool:
        res = []
        for i, r in enumerate(pool.imap(_solve_chunk, args), 1):
            res.append(r)
            print(f"  chunk {i}/{len(args)}  {time.time() - t0:.0f}s", flush=True)
    sol = np.concatenate(res)

    ok = sol[:, 7] == 1
    print(f"solved {ok.sum()}/{len(ok)} within node cap ({100 * ok.mean():.1f}%), "
          f"{time.time() - t0:.0f}s")
    np.savez_compressed(out, boards=boards, movers=movers, plies=plies, cps=cps,
                        hists=hists,
                        signs=sol[:, :7], ok=sol[:, 7])


if __name__ == "__main__":
    main()
