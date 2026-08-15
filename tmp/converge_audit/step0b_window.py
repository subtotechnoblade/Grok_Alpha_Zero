"""Pin production's num_previous_generations from Connect4/log.txt (= run3).

log.txt lists WHICH generations were collected each time.  At gen 5 it collected
gens 5,4,3,2,1 = FIVE files.  Dataloader takes files[-(n_prev+1):] then drops the
oldest, so five files at gen 5 means n_prev = 5, not the 4 that both versions of
Connect4/config.py currently show and that every sweep assumed.

The raw (pre-dedup) position count is the discriminator: it does not depend on
the random game permutation the way the unique count does, only on how many
games each generation contributes.
"""
import numpy as np
from common import D

LOG = {  # gen: (train_raw, test_raw, gens listed in log.txt)
    4: (104890, 11610, [4, 3, 2, 1]),
    5: (123918, 13426, [5, 4, 3, 2, 1]),
}

print("run3, train_decay 0.1 / test_decay 0.01")
for gen, (traw, sraw, gens) in LOG.items():
    print(f"\ngen {gen}: log collected {gens} -> train_raw {traw}, test_raw {sraw}, "
          f"total {traw + sraw}")
    for n_prev in (4, 5, 6):
        d = D.build("run3", gen, n_prev=n_prev, train_decay=0.1, test_decay=0.01,
                    next_gen_holdout=False, verbose=False, seed=0)
        tot = d["train"]["n_raw"] + d["val"]["n_raw"]
        print(f"  n_prev={n_prev}: train_raw {d['train']['n_raw']:>7} "
              f"test_raw {d['val']['n_raw']:>6} total {tot:>7} "
              f"| delta vs log {tot - (traw + sraw):+6d}")
