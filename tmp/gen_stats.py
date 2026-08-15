"""Per-generation outcome rates. game_stats =
[max_actions, total_actions, num_games, p(-1) wins, draws, p(+1) wins]; -1 moves first."""
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import h5py as h5

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = sys.argv[1] if len(sys.argv) > 1 else "Connect4/Grok_Zero_Train"

rows = []
for d in sorted(glob(f"{RUN}/*/"), key=lambda p: int(Path(p).name)):
    f_path = os.path.join(d, "Self_Play_Data.h5")
    if not os.path.exists(f_path):
        continue
    with h5.File(f_path, "r") as f:
        mx, tot, n, w1, dr, w2 = [int(x) for x in f["game_stats"][:]]
    if n == 0:
        continue
    rows.append((int(Path(d).name), n, mx, tot / n, w1 / n, dr / n, w2 / n))

print(f"{'gen':>4} {'games':>6} {'maxlen':>7} {'avglen':>7} "
      f"{'P-1 first':>10} {'draw':>7} {'P+1 second':>11} {'2nd-1st':>8} {'+-2se':>7}")
print("-" * 74)
for g, n, mx, avg, a, d, b in rows:
    se = float(np.sqrt(max(a * (1 - a) + b * (1 - b), 1e-9) / n))
    print(f"{g:>4} {n:>6} {mx:>7} {avg:>7.1f} {a:>9.1%} {d:>6.1%} {b:>10.1%} "
          f"{b-a:>+7.1%} {2*se:>6.1%}")
