"""Inventory: games per generation, positions per game, singleton rate, target stats."""
import numpy as np
import common as C

for run, gens in (("run2", range(0, 20)), ("run3", range(0, 16))):
    print(f"\n=== {run} (q_weight={C.D.Q_WEIGHT[run]}) ===")
    print(f"{'gen':>4} {'games':>6} {'raw pos':>8} {'uniq':>7} {'dup%':>6} "
          f"{'once%':>6} {'meanZ':>7} {'varZ':>7} {'|blend|max':>10} {'plies/game':>10}")
    for g in gens:
        games = C.read_gen(run, g)
        if not games:
            print(f"{g:>4}  (missing)")
            continue
        p = C.pack(games)
        z = np.concatenate([x["z"] for x in games]).ravel()
        nraw = p["n_raw"]
        print(f"{g:>4} {len(games):>6} {nraw:>8} {len(p['X']):>7} "
              f"{100*(1-len(p['X'])/nraw):>5.1f}% {100*np.mean(p['count']==1):>5.1f}% "
              f"{z.mean():>7.4f} {z.var():>7.4f} "
              f"{np.abs(p['y_blend']).max():>10.3f} {nraw/len(games):>10.1f}")
