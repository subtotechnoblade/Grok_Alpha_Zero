"""How does the per-position outcome variance sigma^2 vary with game phase?

The noise-floor estimate in exp_fast section 4 can only be measured on positions seen
>= 4 times in train, and those skew to the opening (mean ply 9.3). Once-only positions
sit at ply ~21-23. If sigma^2 FALLS with ply (deeper positions are more decided), then
extrapolating the opening's sigma^2 to once-only positions OVERSTATES their noise floor
and the honest floor is lower. This measures the trend so the extrapolation is bounded
rather than assumed.
"""
import json
import os
import sys

import numpy as np

import data


def main(run="run2", gen=12):
    W = data.raw_window(run, gen, seed=0)
    out = {"run": run, "gen": gen}
    for target in ("blend", "z"):
        tgt = "z" if target == "z" else "values"
        gt = data.group_stats(W["train_games"], tgt)
        ply = (gt["X"][:, :, :, 3] != 0).sum((1, 2))
        print(f"\n  {target}: within-position outcome variance sigma^2 by ply "
              f"(train groups with the stated minimum occurrence count)")
        rows = []
        for mn in (2, 4, 10):
            line = []
            for lo, hi in ((0, 5), (6, 9), (10, 13), (14, 17), (18, 21), (22, 29), (30, 41)):
                m = (gt["count"] >= mn) & (ply >= lo) & (ply <= hi)
                if m.sum() < 10:
                    line.append(f"ply{lo:2d}-{hi:2d}:    n/a")
                    continue
                v = float(np.nanmean(gt["var"][m]))
                line.append(f"ply{lo:2d}-{hi:2d}: {v:.3f}({int(m.sum())})")
                rows.append({"target": target, "min_count": mn, "ply_lo": lo, "ply_hi": hi,
                             "sigma2": v, "n_groups": int(m.sum())})
            print(f"    count>={mn:2d}: " + "  ".join(line))
        out[target] = rows
        # weighted mean sigma^2 seen by the count>=4 subset vs where once-only rows live
        m4 = gt["count"] >= 4
        print(f"    -> mean ply of the count>=4 groups the floor was measured on: "
              f"{np.average(ply[m4], weights=gt['count'][m4]):.1f}")
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"sigma_{run}_g{gen}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "run2",
         int(sys.argv[2]) if len(sys.argv) > 2 else 12)
