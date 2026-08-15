"""Experiment G: accuracy per second, not accuracy per epoch.

Two separate questions get conflated by "is muon faster":
  1. per step cost   - how long does one optimizer step take?
  2. per second gain  - at a fixed wall clock budget, who is ahead on accuracy?

Part 1 reads the wall clock already recorded in the sweeps. Part 2 replays each arm's
val accuracy curve against elapsed seconds at its own best learning rate.

    python tmp/exp_g_speed.py
"""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BEST = {"adamw": ("sweep", "adamw@0.003"),
        "muon": ("sweep2", "muon@0.1"),
        "muon_buggy": ("sweep", "muon_buggy@0.01"),
        "muon_ortho": ("sweep2", "muon_ortho@0.1"),
        "muon_grokfast": ("sweep", "muon_grokfast@0.03")}


def load(stage):
    with open(os.path.join(ROOT, "tmp", f"results_cifar_{stage}.json")) as f:
        return json.load(f)


def main():
    data = {s: load(s) for s in ("sweep", "sweep2", "sweep3")}

    print("per epoch wall clock at each arm's best lr (includes XLA compile in epoch 1)\n")
    print(f"{'arm':<16} {'lr':>8} {'total s':>9} {'s/epoch':>9} {'best acc':>10} {'acc/min':>9}")
    print("-" * 66)
    rows = {}
    for arm, (stage, key) in BEST.items():
        r = data[stage][key]
        per_ep = r["secs"] / len(r["curve"])
        rows[arm] = r
        print(f"{arm:<16} {key.split('@')[1]:>8} {r['secs']:>9.0f} {per_ep:>9.1f} "
              f"{r['best_acc']:>10.4f} {r['best_acc'] / (r['secs'] / 60):>9.4f}")

    print("\n\nval accuracy vs elapsed seconds (each arm at its own best lr)\n")
    marks = sorted({round((i + 1) * r["secs"] / len(r["curve"]))
                    for r in rows.values() for i in range(len(r["curve"]))})
    header = "".join(f"{m:>8}s" for m in marks[:1] + marks[1:0])
    print(f"{'arm':<16}" + "".join(f"{f'ep{i+1}':>9}" for i in range(5)))
    print("-" * (16 + 45))
    for arm, r in rows.items():
        print(f"{arm:<16}" + "".join(f"{a:>9.4f}" for a in r["curve"]))
    print(f"{'':<16}" + "".join(f"{f'{rows[a][chr(39)]}':>9}" for a in []))

    print("\nsame curves, but on a shared wall clock grid (linear interp between epochs)")
    grid = [60, 120, 180, 240, 300]
    print(f"{'arm':<16}" + "".join(f"{f'{t}s':>9}" for t in grid))
    print("-" * (16 + 45))
    for arm, r in rows.items():
        per_ep = r["secs"] / len(r["curve"])
        line = f"{arm:<16}"
        for t in grid:
            pos = t / per_ep
            if pos < 1:
                line += f"{'-':>9}"
            elif pos >= len(r["curve"]):
                line += f"{max(r['curve']):>9.4f}"     # budget exceeds the run, best so far
            else:
                lo = int(pos) - 1
                frac = pos - int(pos)
                v = r["curve"][lo] + frac * (r["curve"][min(lo + 1, len(r["curve"]) - 1)] - r["curve"][lo])
                line += f"{v:>9.4f}"
        print(line)


if __name__ == "__main__":
    main()
