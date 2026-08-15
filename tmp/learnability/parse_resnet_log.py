"""Recover a ResNet config's curve from resnet_run2.log (the long 4-config run was cut
short once the machine filled with other jobs; its completed configs are still fully
recorded in the log, so nothing measured is thrown away)."""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HDR = re.compile(r"\[(?P<tag>[^\]]+)\] params=(?P<p>\d+) target=(?P<t>\w+) orthograd=(?P<o>\w+)")
EP = re.compile(r"ep\s*(?P<e>\d+) loss=(?P<l>[-\d.]+) train_(?P<t>\w+)=(?P<tr>[-\d.]+) "
                r"val_\w+=(?P<v>[-\d.]+)(?: nxt_\w+=(?P<n>[-\d.]+))?")


def main(log="resnet_run2.log", out="resnet2_run2_g12_blend.json"):
    cfgs, cur = [], None
    for line in open(os.path.join(HERE, log), errors="ignore"):
        h = HDR.search(line)
        if h:
            cur = {"tag": h["tag"], "target": h["t"], "params": int(h["p"]), "curve": []}
            cfgs.append(cur)
            continue
        m = EP.search(line)
        if m and cur is not None:
            cur["curve"].append({"epoch": int(m["e"]), "train_loss": float(m["l"]),
                                 f"train_{m['t']}": float(m["tr"]),
                                 f"val_{m['t']}": float(m["v"]),
                                 **({f"nxt_{m['t']}": float(m["n"])} if m["n"] else {})})
    rows = []
    for c in cfgs:
        if not c["curve"]:
            continue
        t = c["target"]
        best = min(c["curve"], key=lambda r: r[f"val_{t}"])
        rows.append({"tag": c["tag"], "target": t, "best_epoch": best["epoch"],
                     "epochs_run": len(c["curve"]),
                     "val": best[f"val_{t}"], "nxt": best.get(f"nxt_{t}"),
                     "train_at_best": best[f"train_{t}"],
                     "final_train": c["curve"][-1][f"train_{t}"],
                     "final_val": c["curve"][-1][f"val_{t}"],
                     "final_nxt": c["curve"][-1].get(f"nxt_{t}"),
                     "curve": c["curve"]})
        print(f"{c['tag']:28s} [{t}] {len(c['curve'])} epochs, best ep {best['epoch']}: "
              f"train={best[f'train_{t}']:.4f} val={best[f'val_{t}']:.4f} "
              f"nxt={best.get(f'nxt_{t}', float('nan')):.4f}")
    p = os.path.join(HERE, out)
    existing = json.load(open(p)) if os.path.exists(p) else []
    tags = {(r["tag"], r["target"]) for r in existing}
    existing += [r for r in rows if (r["tag"], r["target"]) not in tags]
    with open(p, "w") as f:
        json.dump(existing, f, indent=1)
    print(f"wrote {p} ({len(existing)} configs)")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
