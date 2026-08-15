"""M3b: live MCTS full mode for gens 6 and 12 (completes the root-Q trajectory)."""
import os, sys
ROOT = r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np
import multiprocessing as mp
sys.path.insert(0, os.path.join(ROOT, "tmp", "fresh2_cycle"))
from m3_live_mcts import run_batch

if __name__ == "__main__":
    jobs = [(6, "full", 16, 9000 + i) for i in range(3)] + \
           [(12, "full", 16, 9100 + i) for i in range(3)]
    with mp.Pool(3) as pool:
        results = pool.map(run_batch, jobs)
    agg = {}
    for gen, mode, dists, qsv, am in results:
        agg.setdefault(gen, [[], [], []])
        agg[gen][0].append(dists); agg[gen][1].append(qsv); agg[gen][2].extend(am)
    for gen, (dl, ql, am) in sorted(agg.items()):
        dists = np.concatenate(dl); qsv = np.concatenate(ql)
        md, mq = dists.mean(0), np.nanmean(qsv, 0)
        amc = np.bincount(am, minlength=7)
        print(f"\n=== gen {gen} mode=full ({len(dists)} runs) ===")
        print("            " + "".join(f"  col{c}  " for c in range(7)))
        print("live visits " + "".join(f" {md[c]:6.3f} " for c in range(7)))
        print("live root Q " + "".join(f" {mq[c]:6.3f} " for c in range(7)))
        print("argmax count" + "".join(f" {amc[c]:6d} " for c in range(7)))
