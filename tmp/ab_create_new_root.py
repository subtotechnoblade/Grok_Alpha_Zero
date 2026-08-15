"""Does create_new_root cause the player +1 bias? Same net, same everything else.

create_new_root=True rebuilds the root through create_expand_root() on EVERY
move. That path also runs get_terminal_actions() at the root every move, which
the tree-reuse path never did. If either introduces a player asymmetry, the two
arms will disagree on the p-1 / p+1 split.

  usage: ab_create_new_root.py [games] [workers]
"""
import os
import sys
import shutil
import time
from pathlib import Path

ROOT = Path(r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

SRC_MODEL = ROOT / "Connect4" / "Grok_Zero_Train" / "7" / "model.onnx"
WORK = ROOT / "tmp" / "ab_root" / "Grok_Zero_Train"


def run(create_new_root, games, workers):
    import h5py as h5
    from Connect4.game import Connect4
    from Connect4.config import train_config, build_config, optimizer_config
    from Self_Play import run_self_play
    from pipeline import Make_Dataset_File

    cfg = train_config.replace(games_per_generation=games, num_workers=workers,
                               create_new_root=create_new_root)

    shutil.rmtree(WORK.parent, ignore_errors=True)
    gen1 = WORK / "1"
    gen1.mkdir(parents=True)
    shutil.copy2(SRC_MODEL, gen1 / "model.onnx")
    folder = gen1.as_posix()
    Make_Dataset_File(folder)

    t0 = time.perf_counter()
    run_self_play(Connect4, (build_config, cfg, optimizer_config), folder)
    wall = time.perf_counter() - t0

    with h5.File(gen1 / "Self_Play_Data.h5", "r") as f:
        ma, ta, n, w1, dr, w2 = f["game_stats"][:]
    shutil.rmtree(WORK.parent, ignore_errors=True)
    return dict(n=int(n), pm1=int(w1) / int(n), draw=int(dr) / int(n),
                p1=int(w2) / int(n), moves=int(ta) / int(n), wall=wall)


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    print(f"gen-7 net | {games} games per arm | {workers} workers\n")
    rows = []
    for cnr in (True, False):
        print(f"--- create_new_root = {cnr} ---", flush=True)
        r = run(cnr, games, workers)
        r["cnr"] = cnr
        rows.append(r)
        print(f"    {r['wall']:.0f}s\n", flush=True)

    print("=" * 68)
    print(f"{'create_new_root':>16} {'games':>6} {'p-1 win':>8} {'draw':>7} "
          f"{'p+1 win':>8} {'avg mv':>7}")
    for r in rows:
        print(f"{str(r['cnr']):>16} {r['n']:>6} {r['pm1']:>8.3f} {r['draw']:>7.3f} "
              f"{r['p1']:>8.3f} {r['moves']:>7.1f}")
    print()
    print("training run gen 6 for reference: p-1 0.306 / draw 0.140 / p+1 0.554")
    print("a fair implementation should NOT show a large p+1 edge in either arm")


if __name__ == "__main__":
    main()
