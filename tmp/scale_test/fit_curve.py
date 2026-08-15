"""Fit the learning curves and extrapolate: how many games to reach k x floor?

Two fits per split, because the answer depends entirely on which is right:
  POWER      L(N) = a * N^-b            (no asymptote - the optimistic reading;
                                         data alone eventually gets you anywhere)
  POWER+ASY  L(N) = a * N^-b + c        (an irreducible c that data cannot buy off)

Reported with the residuals of both so the reader can see which the data supports.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def fit_power(N, L):
    """least squares in log space -> L = a N^-b"""
    A = np.vstack([np.ones_like(N), -np.log(N)]).T
    coef, *_ = np.linalg.lstsq(A, np.log(L), rcond=None)
    a, b = np.exp(coef[0]), coef[1]
    pred = a * N ** -b
    return a, b, float(np.sqrt(np.mean((pred - L) ** 2)))


def fit_power_asy(N, L):
    """grid search c, power-fit the residual"""
    best = None
    for c in np.linspace(0.0, 0.995 * L.min(), 400):
        try:
            a, b, r = fit_power(N, L - c)
        except Exception:
            continue
        pred = a * N ** -b + c
        rms = float(np.sqrt(np.mean((pred - L) ** 2)))
        if best is None or rms < best[3]:
            best = (a, b, c, rms)
    return best


def need(a, b, c, target, N0):
    """games needed for a N^-b + c == target"""
    if target <= c:
        return float("inf")
    return float(((target - c) / a) ** (-1.0 / b))


def analyse(name, N, L, floor, N0):
    print(f"\n  --- {name} ---")
    for n, l in zip(N, L):
        print(f"      N={n:>6}  loss={l:.4f}  = {l/floor:>5.1f}x floor")
    a, b, r = fit_power(N, L)
    print(f"      POWER      L = {a:.4f} * N^-{b:.4f}          rms {r:.5f}")
    aa, bb, cc, rr = fit_power_asy(N, L)
    print(f"      POWER+ASY  L = {aa:.4f} * N^-{bb:.4f} + {cc:.4f}   rms {rr:.5f}")
    print(f"      {'target':>16} {'power: games':>16} {'x current':>10} | "
          f"{'+asy: games':>16} {'x current':>10}")
    for k, lab in ((3.0, "3x floor"), (2.0, "2x floor"), (1.0, "1x floor")):
        t = k * floor
        n1 = need(a, b, 0.0, t, N0)
        n2 = need(aa, bb, cc, t, N0)
        f1 = f"{n1:.3g}" if np.isfinite(n1) else "unreachable"
        f2 = f"{n2:.3g}" if np.isfinite(n2) else "UNREACHABLE"
        x1 = f"{n1/N0:.3g}x" if np.isfinite(n1) else "-"
        x2 = f"{n2/N0:.3g}x" if np.isfinite(n2) else "-"
        print(f"      {lab+f' ({t:.4f})':>16} {f1:>16} {x1:>10} | {f2:>16} {x2:>10}")
    return dict(power=dict(a=a, b=b, rms=r),
                asy=dict(a=aa, b=bb, c=cc, rms=rr), floor=floor)


def main(path):
    d = json.load(open(path))
    rows, fl = d["rows"], d["floors"]
    print(f"### curve fit on {os.path.basename(path)}   "
          f"floors val {fl['val']:.4f} nxt {fl['nxt']:.4f}")
    print(f"    const0  val {d['const0']['val']:.4f} nxt {d['const0']['nxt']:.4f}")
    out = {}
    for model in sorted({r["model"] for r in rows}):
        sizes = sorted({r["n_games"] for r in rows if r["model"] == model})
        N = np.array(sizes, dtype=float)
        N0 = float(max(sizes))
        for split in ("val", "nxt"):
            L = np.array([np.mean([r[split] for r in rows
                                   if r["model"] == model and r["n_games"] == n])
                          for n in sizes])
            sd = np.array([np.std([r[split] for r in rows
                                   if r["model"] == model and r["n_games"] == n])
                           for n in sizes])
            print(f"\n  {model} {split}: seed sd per size = " +
                  " ".join(f"{x:.4f}" for x in sd))
            out[f"{model}_{split}"] = analyse(f"{model} / {split}", N, L, fl[split], N0)
    with open(os.path.join(HERE, "fit_" + os.path.basename(path)), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else os.path.join(HERE, "exp1_run2_g12_blend.json"))
