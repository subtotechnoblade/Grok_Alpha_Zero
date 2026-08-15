"""M10: match the split config actually used by Connect4/log.txt and compare its
observed val_value_loss against the measured floor.

The log's dedup line at gen 5 reads train 123918 -> 67710, test 13426 -> 10001.
123918 raw train samples over a 4-gen window of 500 games (~16k plies x2 aug =
~128k) implies train_percent 1.0 with NO decay, and test ~10% flat.  So rerun the
floor with train_decay=0, test_decay=0 to line the two up.

The log is from a LATER run than the archives (post weight-fix, q_weight 0.4 by
inference: its epoch-1 val_value_loss 0.4720 ~ E[t^2]=0.478 measured on run3 g0).
So this is a like-for-like comparison across runs, not the same run - labelled.
"""
import numpy as np, m4_split as M

FLAT = dict(num_previous_generations=4, train_percent=1.0, train_decay=0.0,
            test_percent=0.1, test_decay=0.0)

for g in (3, 4, 5):
    M.report("run3", g, target="t", cfg=FLAT, label="[flat decays, matches log.txt]")
