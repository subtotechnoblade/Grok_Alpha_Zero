"""Merge every results_*.json / resnet_*.json / trivial_*.json into one table."""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

ORDER = [
    "constant: predict 0",
    "constant: train weighted mean",
    "per-ply mean",
    "per-(ply,side) mean",
    "per-(ply,side,playable-threats) mean",
    "ridge [raw] (168d)",
    "ridge [rel] (127d)",
    "ridge [hand] (30d)",
    "ridge [hand+rel] (157d)",
    "kNN hamming k=1", "kNN hamming k=3", "kNN hamming k=10", "kNN hamming k=30",
    "RandomForest [hand]",
    "HistGradientBoosting [hand]",
    "HistGradientBoosting [hand+rel]",
    "XGBoost [hand+rel]",
    "MLP (32, 32) [hand]",
    "MLP (32, 32) [hand+rel]",
    "MLP (64, 64) [rel]",
    "ResNet arch + plain Adam",
    "ResNet prod (Muon only)",
    "ResNet prod +OrthoGrad",
    "ORACLE lookup (train group means)",
]


def load(run, gen):
    rows = {}
    for pat in (f"results_{run}_g{gen}.json", f"trivial_{run}_g{gen}.json"):
        p = os.path.join(HERE, pat)
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        for r in (j["rows"] if isinstance(j, dict) else j):
            rows.setdefault(r["model"], {})[r["target"]] = r
    for p in (glob.glob(os.path.join(HERE, f"resnet_{run}_g{gen}.json")) +
              glob.glob(os.path.join(HERE, f"resnet2_{run}_g{gen}_*.json"))):
        for r in json.load(open(p)):
            rows.setdefault(r["tag"], {})[r["target"]] = {
                "val": r["val"], "nxt": r["nxt"],
                "note": f"best ep {r['best_epoch']}/{r.get('epochs_run', len(r['curve']))}, "
                        f"train={r['train_at_best']:.3f}"}
    return rows


def fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "  -   "


def main(run="run2", gen=12):
    rows = load(run, gen)
    print(f"\n{'='*104}\n{run} window gen {gen-3}-{gen}  (production 4-gen window)   "
          f"val = production in-window holdout | nxt = whole next generation, untouched\n{'='*104}")
    print(f"{'model':40s} | {'target = stored blend':>23s} | {'target = pure z':>23s}")
    print(f"{'':40s} | {'val':>10s} {'nxt':>10s} | {'val':>10s} {'nxt':>10s}   note")
    print("-" * 104)
    seen = set()
    for name in ORDER + sorted(k for k in rows if k not in ORDER):
        if name in seen or name not in rows:
            continue
        seen.add(name)
        b = rows[name].get("blend", {})
        z = rows[name].get("z", {})
        note = b.get("note") or z.get("note") or ""
        print(f"{name:40s} | {fmt(b.get('val')):>10s} {fmt(b.get('nxt')):>10s} | "
              f"{fmt(z.get('val')):>10s} {fmt(z.get('nxt')):>10s}   {note}")
    # relative to floor
    f_b_val = rows.get("constant: predict 0", {}).get("blend", {}).get("val")
    f_b_nxt = rows.get("constant: predict 0", {}).get("blend", {}).get("nxt")
    print("-" * 104)
    print(f"floors: blend val={fmt(f_b_val)} nxt={fmt(f_b_nxt)}; "
          f"z val={fmt(rows.get('constant: predict 0',{}).get('z',{}).get('val'))} "
          f"nxt={fmt(rows.get('constant: predict 0',{}).get('z',{}).get('nxt'))}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12)
    else:
        for f in sorted(glob.glob(os.path.join(HERE, "results_*.json"))):
            b = os.path.basename(f)[len("results_"):-len(".json")]
            run, gen = b.rsplit("_g", 1)
            main(run, int(gen))
