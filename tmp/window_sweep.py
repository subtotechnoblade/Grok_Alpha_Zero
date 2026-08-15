"""How many generations should the window hold? Same harness, window LENGTH varies.

The decay sweep showed deployment error tracks n_train, not recency. So the question
becomes: does adding older generations keep helping, or does staleness eventually bite?
Shape held at production's flat 0.1; only n_prev changes.
"""
import json, os, sys, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "learnability")); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, tensorflow as tf
import data as D
from valmetrics import ply_of, wmse, position_floor
from weight_sweep import build_and_compile

RUN, GEN, TARGET, EPOCHS = "run2", 12, "y_blend", 10
SEEDS = (0, 1)
floors = json.load(open(os.path.join(HERE, "learnability", f"floorfull_{RUN}_g{GEN}.json")))
base = D.get(RUN, GEN)
ev = {}
for sp in ("nxt", "val"):
    s = base[sp]; ply = ply_of(s["X"])
    ev[sp] = dict(X=s["X"].astype(np.float32), y=s[TARGET], w=s["w"], ply=ply,
                  floor=position_floor(ply, s["count"], floors, "blend"))
nfl = float(np.average(ev["nxt"]["floor"], weights=ev["nxt"]["w"]))
vfl = float(np.average(ev["val"]["floor"], weights=ev["val"]["w"]))
m_end = ev["nxt"]["ply"] >= 22
efl = float(np.average(ev["nxt"]["floor"][m_end], weights=ev["nxt"]["w"][m_end]))
print(f"{RUN} gen {GEN} | flat decay 0.1 | {EPOCHS} epochs | seeds {SEEDS}")
print(f"nxt floor {nfl:.4f}  const-0 {wmse(ev['nxt']['y'], np.zeros(len(ev['nxt']['y'])), ev['nxt']['w']):.4f}\n")
print(f"{'window':<12} {'n_train':>8} | {'NXT':>8} {'nxt/fl':>7} {'nxt end':>8} | {'VAL':>8} {'val/fl':>7}")
print("-" * 70)
out = {}
for n_prev in (1, 2, 4, 6, 8, 11):
    d = D.build(RUN, GEN, n_prev=n_prev, train_decay=0.1, verbose=False, next_gen_holdout=False)
    tr = d["train"]
    X = tr["X"].astype(np.float32); P = tr["policy"].astype(np.float32)
    V = tr[TARGET].astype(np.float32)[:, None]; wf = tr["w"].astype(np.float32)
    acc = {"nxt": [], "nxt_end": [], "val": []}
    for seed in SEEDS:
        tf.keras.utils.set_random_seed(seed)
        mdl = build_and_compile()
        mdl.fit(X, (P, V), sample_weight=(wf, wf), batch_size=1024, epochs=EPOCHS, verbose=0, shuffle=True)
        for sp in ("nxt", "val"):
            e = ev[sp]
            pv = mdl.predict(e["X"], batch_size=4096, verbose=0)[1][:, 0].astype(np.float64)
            acc[sp].append(wmse(e["y"], pv, e["w"]))
            if sp == "nxt":
                acc["nxt_end"].append(wmse(e["y"][m_end], pv[m_end], e["w"][m_end]))
    r = {k: float(np.mean(v)) for k, v in acc.items()}; r["n_train"] = int(len(X))
    r["sd"] = float(np.std(acc["nxt"])); out[n_prev] = r
    print(f"{n_prev:<12} {r['n_train']:>8} | {r['nxt']:>8.4f} {r['nxt']/nfl:>6.1f}x "
          f"{r['nxt_end']:>8.4f} | {r['val']:>8.4f} {r['val']/vfl:>6.1f}x", flush=True)
print("\nseed sd on NXT: " + ", ".join(f"{v['sd']:.4f}" for v in out.values()))
json.dump(out, open(os.path.join(HERE, f"window_sweep_{RUN}_g{GEN}.json"), "w"), indent=1)
