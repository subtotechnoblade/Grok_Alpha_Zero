"""STEP 3 - re-run the decisive arms at BOTH fit levels.

Arm families (the two the owner's sweeps drew conclusions from):
  weight   sample-weighting scheme: count / uniform / cap20
  window   replay-window LENGTH: n_prev 1 / 4 / 11

Fit levels are named in FITS below; every arm is run at every fit level with the
same seeds, so an ordering change between levels is the undertraining artifact
the audit is looking for.

Evaluation splits come from ONE canonical build (n_prev=4, production decay) so
every arm is scored on identical data; only the TRAIN side changes.

usage: step3_arms.py <run> <gen> <family: weight|window> <fit names, comma sep>
"""
import sys
import numpy as np
import tensorflow as tf

from common import (D, cbuild, build_and_compile, prep_eval, floors_for, arch_of,
                    weights_path, value_pred, wmse, dump, BATCH, position_floor, ply_of)

RUN = sys.argv[1] if len(sys.argv) > 1 else "run3"
GEN = int(sys.argv[2]) if len(sys.argv) > 2 else 12
FAMILY = sys.argv[3] if len(sys.argv) > 3 else "weight"
TARGET = "y_blend"
TRAIN_DECAY, TEST_DECAY = 0.1, 0.01
SEEDS = (0, 1, 2)

# name -> dict(init, lr, wd, epochs, arch).  wd None = Muon's DEFAULT 0.004, which
# is what the sweeps silently got (the working tree's config.py comments
# weight_decay out); 1e-4 is git HEAD's config.py = what both archived runs used.
# arch None = the run's own production architecture; "current" on run2 is the net
# the sweeps actually trained.
FITS = {
    "sweep":     dict(init="scratch", lr=3e-4, wd=None, epochs=10, arch="current"),
    "sweep_old": dict(init="scratch", lr=3e-4, wd=None, epochs=10, arch=None),
    "prod":      dict(init="warm",    lr=3e-4, wd=1e-4, epochs=8,  arch=None),
    "converged": dict(init="warm",    lr=3e-4, wd=1e-4, epochs=40, arch=None),
}
FIT_NAMES = (sys.argv[4].split(",") if len(sys.argv) > 4 else list(FITS))


def norm1(w):
    return w * (len(w) / w.sum())


def weight_arms(tr, floors):
    n = tr["count"].astype(np.float64)
    return {"count": norm1(n),
            "uniform": norm1(np.ones_like(n)),
            "cap20": norm1(np.minimum(n, 20.0))}


def main():
    base = cbuild(RUN, GEN, n_prev=4, train_decay=TRAIN_DECAY, test_decay=TEST_DECAY,
                  next_gen_holdout=True, verbose=True)
    floors = floors_for(RUN, GEN)
    ev = prep_eval(base, ("val", "nxt"), TARGET, floors, "blend")
    arch = arch_of(RUN)
    warm_from = weights_path(RUN, GEN)

    # ---- build the arms: (name, train_dict, weight_vector) ----
    arms = []
    if FAMILY == "weight":
        tr = base["train"]
        for name, w in weight_arms(tr, floors).items():
            arms.append((name, tr, w))
    elif FAMILY == "window":
        for n_prev in (1, 4, 11):
            d = cbuild(RUN, GEN, n_prev=n_prev, train_decay=TRAIN_DECAY,
                       test_decay=TEST_DECAY, next_gen_holdout=True)
            tr = d["train"]
            arms.append((f"n_prev={n_prev}", tr, tr["w"].astype(np.float64)))
    else:
        raise SystemExit(f"unknown family {FAMILY}")

    m_end = ev["nxt"]["ply"] >= 22
    nfl = float(np.average(ev["nxt"]["floor"], weights=ev["nxt"]["w"]))
    vfl = float(np.average(ev["val"]["floor"], weights=ev["val"]["w"]))
    print(f"\n{RUN} gen {GEN} | family {FAMILY} | arch {arch} | seeds {SEEDS}")
    print(f"  nxt floor {nfl:.4f}  const-0 "
          f"{wmse(ev['nxt']['y'], np.zeros(len(ev['nxt']['y'])), ev['nxt']['w']):.4f}")
    print(f"  val floor {vfl:.4f}  const-0 "
          f"{wmse(ev['val']['y'], np.zeros(len(ev['val']['y'])), ev['val']['w']):.4f}")

    results = {}
    for fit_name in FIT_NAMES:
        cfg = FITS[fit_name]
        print(f"\n{'='*92}\nFIT '{fit_name}': {cfg}\n{'='*92}")
        hdr = (f"{'arm':<12} {'n_train':>8} | {'TRAIN':>8} {'sd':>6} | {'VAL':>8} {'sd':>6} "
               f"{'v/fl':>6} | {'NXT':>8} {'sd':>6} {'n/fl':>6} | {'NXT end':>8}")
        print(hdr + "\n" + "-" * len(hdr))
        for name, tr, w in arms:
            X = tr["X"].astype(np.float32)
            P = tr["policy"].astype(np.float32)
            V = tr[TARGET].astype(np.float32)[:, None]
            wf = w.astype(np.float32)
            acc = {k: [] for k in ("train", "val", "nxt", "nxt_end")}
            a = cfg.get("arch") or arch
            if cfg["init"] == "warm" and a != arch:
                raise SystemExit(f"fit {fit_name}: no {a}-arch checkpoint for {RUN}")
            for seed in SEEDS:
                tf.keras.utils.set_random_seed(seed)
                model = build_and_compile(lr=cfg["lr"], arch=a, weight_decay=cfg["wd"],
                                          warm=(warm_from if cfg["init"] == "warm" else None))
                model.fit(X, (P, V), sample_weight=(wf, wf), batch_size=BATCH,
                          epochs=cfg["epochs"], verbose=0, shuffle=True)
                # train fit, measured with the arm's OWN weights (that is the
                # objective the arm was actually optimising)
                acc["train"].append(wmse(tr[TARGET], value_pred(model, X), w))
                for sp in ("val", "nxt"):
                    e = ev[sp]
                    pv = value_pred(model, e["Xf"])
                    acc[sp].append(wmse(e["y"], pv, e["w"]))
                    if sp == "nxt":
                        acc["nxt_end"].append(wmse(e["y"][m_end], pv[m_end], e["w"][m_end]))
            r = {k: float(np.mean(v)) for k, v in acc.items()}
            r.update({f"{k}_sd": float(np.std(v)) for k, v in acc.items()})
            r["n_train"] = int(len(X))
            r["raw"] = {k: [float(x) for x in v] for k, v in acc.items()}
            results[f"{fit_name}|{name}"] = r
            print(f"{name:<12} {r['n_train']:>8} | {r['train']:>8.4f} {r['train_sd']:>6.4f} | "
                  f"{r['val']:>8.4f} {r['val_sd']:>6.4f} {r['val']/vfl:>5.1f}x | "
                  f"{r['nxt']:>8.4f} {r['nxt_sd']:>6.4f} {r['nxt']/nfl:>5.1f}x | "
                  f"{r['nxt_end']:>8.4f}", flush=True)
            dump(f"step3_{FAMILY}_{RUN}_g{GEN}.json",
                 {"run": RUN, "gen": GEN, "family": FAMILY, "seeds": list(SEEDS),
                  "fits": {k: {kk: vv for kk, vv in v.items()} for k, v in FITS.items()},
                  "nxt_floor": nfl, "val_floor": vfl, "results": results})


if __name__ == "__main__":
    main()
