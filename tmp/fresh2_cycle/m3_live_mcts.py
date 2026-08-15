"""M3: live MCTS at the empty board, self-play conditions, per gen.

Modes:
  full   : dirichlet ON (alpha 1.33, eps 0.25), value head ON   <- replicates self play
  nonoise: dirichlet OFF, value ON (deterministic-ish)
  noval  : dirichlet ON, value head silenced (returns 0.0)      <- prior+noise+U only
  novalnn: dirichlet OFF, value silenced

Reports mean root visit share per column, per-run argmax counts, and the mean
root Q (winrate, opener's perspective) per column after 375 sims.
"""
import os, sys
ROOT = r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import multiprocessing as mp


def run_batch(args):
    gen, mode, n_runs, seed = args
    import numpy as np
    import onnxruntime as rt
    rt.set_default_logger_severity(4)
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    from Connect4.game import Connect4
    from MCTS import MCTS

    np.random.seed(seed)
    so = rt.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = rt.InferenceSession(f"Connect4/Grok_Zero_Train/{gen}/model.onnx",
                               sess_options=so, providers=["CPUExecutionProvider"])
    use_dirichlet = mode in ("full", "noval")
    silence_value = mode in ("noval", "novalnn")

    dists, qs, argmaxes = [], [], []
    for _ in range(n_runs):
        g = Connect4()
        tree = MCTS(game=g, session=sess, use_njit=None,
                    c_puct_init=1.5,
                    use_dirichlet=use_dirichlet,
                    dirichlet_alpha=1.33, dirichlet_epsilon=0.25,
                    tau=1.0, fast_find_win=False)
        if silence_value:
            orig = tree._compute_outputs
            tree._compute_outputs = lambda inputs, depth=0: (orig(inputs, depth)[0], 0.0)
            # root was already expanded in __init__ with the real value head;
            # rebuild it so the silenced head applies from sim 1
            tree.create_expand_root()
        _, probs = tree.run(iteration_limit=375, use_bar=False)
        dist = np.zeros(7); q = np.full(7, np.nan)
        for row in probs:
            a = int(np.ravel(row[0])[0]) if np.ndim(row[0]) else int(row[0])
            dist[a] = row[1]; q[a] = row[2]
        dists.append(dist); qs.append(q); argmaxes.append(int(np.argmax(dist)))
    return gen, mode, np.array(dists), np.array(qs), argmaxes


if __name__ == "__main__":
    jobs = []
    seed = 1234
    for gen in (7, 8, 9, 10, 11):
        for chunk in range(3):
            jobs.append((gen, "full", 16, seed)); seed += 1
    for chunk in range(3):
        jobs.append((11, "noval", 16, seed)); seed += 1
    jobs.append((11, "nonoise", 3, seed)); seed += 1
    jobs.append((11, "novalnn", 3, seed)); seed += 1
    jobs.append((8, "nonoise", 3, seed)); seed += 1
    jobs.append((8, "noval", 16, seed)); seed += 1

    with mp.Pool(3) as pool:
        results = pool.map(run_batch, jobs)

    agg = {}
    for gen, mode, dists, qs, am in results:
        k = (gen, mode)
        if k not in agg:
            agg[k] = [[], [], []]
        agg[k][0].append(dists); agg[k][1].append(qs); agg[k][2].extend(am)

    stored = {7: [0.017,0.036,0.048,0.277,0.586,0.024,0.013],
              8: [0.008,0.419,0.138,0.160,0.141,0.124,0.010],
              9: [0.009,0.403,0.023,0.153,0.040,0.339,0.032],
              10:[0.007,0.307,0.011,0.072,0.018,0.577,0.008],
              11:[0.014,0.139,0.010,0.089,0.020,0.712,0.015]}

    out = {}
    for (gen, mode), (dl, ql, am) in sorted(agg.items()):
        dists = np.concatenate(dl); qs = np.concatenate(ql)
        mean_d = dists.mean(0)
        mean_q = np.nanmean(qs, 0)
        amc = np.bincount(am, minlength=7)
        out[f"{gen}_{mode}"] = mean_d
        print(f"\n=== gen {gen}  mode={mode}  ({len(dists)} runs) ===")
        print("            " + "".join(f"  col{c}  " for c in range(7)))
        print("live visits " + "".join(f" {mean_d[c]:6.3f} " for c in range(7)))
        if gen in stored and mode == "full":
            print("stored data " + "".join(f" {stored[gen][c]:6.3f} " for c in range(7)))
        print("live root Q " + "".join(f" {mean_q[c]:6.3f} " for c in range(7)))
        print("argmax count" + "".join(f" {amc[c]:6d} " for c in range(7)))
    np.savez(r"tmp\fresh2_cycle\live_mcts.npz", **out)
