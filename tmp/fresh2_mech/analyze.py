"""Aggregate result CSVs: win rates with 95% CI + length-bucket breakdown.

usage: analyze.py <label>=<csv glob> [<label>=<glob> ...]
"""
import glob
import math
import sys
from pathlib import Path

HERE = Path(r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero\tmp\fresh2_mech")

BUCKETS = [(7, 24, "<25"), (25, 35, "25-35"), (36, 41, "36-41"), (42, 42, "42")]


def load(pattern):
    rows = []
    for p in glob.glob(str(HERE / pattern)):
        with open(p) as f:
            next(f)
            for line in f:
                w, l = line.strip().split(",")
                rows.append((int(w), int(l)))
    return rows


def ci(p, n):
    if n == 0:
        return 0.0
    return 1.96 * math.sqrt(max(p * (1 - p), 1e-12) / n)


def report(label, rows):
    n = len(rows)
    if n == 0:
        print(f"{label}: NO DATA")
        return
    w1 = sum(1 for w, _ in rows if w == -1)
    w2 = sum(1 for w, _ in rows if w == 1)
    dr = n - w1 - w2
    avg = sum(l for _, l in rows) / n
    p1, p2 = w1 / n, w2 / n
    print(f"\n== {label} ==  n={n}  avg_len={avg:.1f}")
    print(f"  p-1 win {p1:.3f} +/-{ci(p1, n):.3f} | draw {dr / n:.3f} | "
          f"p+1 win {p2:.3f} +/-{ci(p2, n):.3f} | tilt (p+1 - p-1) {p2 - p1:+.3f}")
    print(f"  {'bucket':>7} {'games':>6} {'p-1':>6} {'draw':>6} {'p+1':>6} {'p+1 share of decided':>21}")
    for lo, hi, name in BUCKETS:
        sub = [(w, l) for w, l in rows if lo <= l <= hi]
        m = len(sub)
        if m == 0:
            print(f"  {name:>7} {0:>6}")
            continue
        b1 = sum(1 for w, _ in sub if w == -1)
        b2 = sum(1 for w, _ in sub if w == 1)
        bd = m - b1 - b2
        dec = b1 + b2
        share = b2 / dec if dec else float("nan")
        print(f"  {name:>7} {m:>6} {b1 / m:>6.3f} {bd / m:>6.3f} {b2 / m:>6.3f} "
              f"{share:>14.3f} ({b2}/{dec})")


for arg in sys.argv[1:]:
    label, pattern = arg.split("=", 1)
    report(label, load(pattern))
