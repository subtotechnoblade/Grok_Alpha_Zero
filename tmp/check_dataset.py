import sys

import h5py as h5

for path in sys.argv[1:]:
    try:
        with h5.File(path, "r") as f:
            n = int(f["game_stats"][2])
            triples = (len(f.keys()) - 1) // 3
            print(f"{path}\n  games {n} | dataset triples {triples} | "
                  f"aug {triples // n if n else '?'} | game_stats {list(f['game_stats'][:])}")
    except Exception as e:
        print(f"{path}\n  UNREADABLE {type(e).__name__}: {e}")
