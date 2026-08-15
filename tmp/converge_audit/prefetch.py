"""CPU-only: warm the npz cache for every window step3 will need."""
from common import cbuild

JOBS = [
    ("run2", 12, 1), ("run2", 12, 4), ("run2", 12, 11),
    ("run3", 12, 4), ("run3", 12, 5), ("run3", 12, 1), ("run3", 12, 11),
]
for run, gen, n_prev in JOBS:
    d = cbuild(run, gen, n_prev=n_prev, train_decay=0.1, test_decay=0.01,
               next_gen_holdout=True, verbose=False)
    print(f"{run} g{gen} n_prev={n_prev}: train {len(d['train']['X'])} "
          f"val {len(d['val']['X'])} nxt {len(d.get('nxt', {}).get('X', []))}", flush=True)
