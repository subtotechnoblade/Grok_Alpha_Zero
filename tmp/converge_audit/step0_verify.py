"""STEP 0 - prove the harness reproduces production before trusting any number.

Three checks:
  A. Connect4/log.txt prints `Dedup: train A -> B, test C -> D` for gens 0-5.
     Rebuild the SAME windows with tmp/learnability/data.py and see whether the
     counts match.  This simultaneously identifies which archive log.txt belongs
     to and pins production's train_decay / test_decay.
  B. Which archived model.weights.h5 loads into which build_model.
  C. The shipped production model evaluated on the reconstructed splits ->
     the reference "production fit" that everything else is measured against.
"""
import sys
import numpy as np

from common import (D, build_and_compile, weights_path, wmse, value_pred, dump, ROOT)

# `Dedup: train X -> Y (..% removed), test Z -> W`  scraped from Connect4/log.txt
LOG = {  # gen: (train_raw, train_uniq, test_raw, test_uniq, n_files)
    0: (28678, 24960, 3200, 2931, 1),
    1: (55244, 45608, 6358, 5729, 2),
    2: (56218, 41246, 5998, 5142, 2),
    3: (81040, 52570, 9154, 7482, 3),
}


def check_a():
    print("=" * 78)
    print("A. does data.py reproduce Connect4/log.txt's dedup counts?")
    print("=" * 78)
    for run in ("run3", "run2"):
        print(f"\n--- {run} ---")
        for gen, (traw, tuniq, sraw, suniq, nf) in LOG.items():
            for td, sd in ((0.1, 0.01), (0.2, 0.02)):
                try:
                    d = D.build(run, gen, n_prev=4, train_decay=td, test_decay=sd,
                                next_gen_holdout=False, verbose=False, seed=0)
                except Exception as e:
                    print(f"  gen {gen} decay {td}: FAILED {e}")
                    continue
                tr, va = d["train"], d["val"]
                hit = (tr["n_raw"] == traw and len(tr["X"]) == tuniq)
                print(f"  gen {gen}  decay {td}/{sd}:  train {tr['n_raw']:>6} -> {len(tr['X']):>6}"
                      f"   test {va['n_raw']:>5} -> {len(va['X']):>5}"
                      f"   | log: {traw} -> {tuniq}, {sraw} -> {suniq}"
                      f"   {'<== RAW MATCH' if tr['n_raw'] == traw else ''}"
                      f"{' + UNIQ MATCH' if hit else ''}")


def check_b():
    print("\n" + "=" * 78)
    print("B. which archived weights load into which build_model?")
    print("=" * 78)
    for run, gen in (("run2", 13), ("run3", 13)):
        for arch in ("current", "old"):
            try:
                m = build_and_compile(arch=arch, warm=weights_path(run, gen))
                n = sum(int(np.prod(w.shape)) for w in m.weights)
                print(f"  {run} gen{gen} weights -> arch={arch:<8} OK   ({n} params)")
                del m
            except Exception as e:
                print(f"  {run} gen{gen} weights -> arch={arch:<8} FAIL ({type(e).__name__}: "
                      f"{str(e).splitlines()[0][:110]})")


def check_c(run, gen, target, train_decay, test_decay):
    """Production's own shipped model, on the reconstructed splits."""
    from common import prep_eval, floors_for, arch_of
    print("\n" + "=" * 78)
    print(f"C. shipped production model {run} gen{gen+1} on {run} gen{gen}'s window "
          f"(decay {train_decay}/{test_decay}, target {target})")
    print("=" * 78)
    d = D.build(run, gen, n_prev=4, train_decay=train_decay, test_decay=test_decay,
                next_gen_holdout=True, verbose=True, seed=0)
    floors = floors_for(run, gen)
    tgt = "blend" if target == "y_blend" else "z"
    ev = prep_eval(d, ("train", "val", "nxt"), target, floors, tgt)

    m = build_and_compile(arch=arch_of(run), warm=weights_path(run, gen + 1))
    out = {}
    for sp, e in ev.items():
        pv = value_pred(m, e["Xf"])
        out[sp] = dict(wmse=wmse(e["y"], pv, e["w"]),
                       floor=float(np.average(e["floor"], weights=e["w"])),
                       const0=wmse(e["y"], np.zeros(len(e["y"])), e["w"]),
                       n=int(len(e["y"])))
        print(f"  {sp:<6} n={out[sp]['n']:>6}  value wMSE {out[sp]['wmse']:.4f}   "
              f"floor {out[sp]['floor']:.4f} ({out[sp]['wmse']/out[sp]['floor']:.1f}x)   "
              f"const-0 {out[sp]['const0']:.4f}")
    # also the warm-start point itself: the model that PLAYED gen `gen`
    m0 = build_and_compile(arch=arch_of(run), warm=weights_path(run, gen))
    print("  --- the warm-start checkpoint (model that generated gen "
          f"{gen}'s games, trained on the window ending at gen {gen-1}) ---")
    for sp, e in ev.items():
        pv = value_pred(m0, e["Xf"])
        out[f"warmstart_{sp}"] = wmse(e["y"], pv, e["w"])
        print(f"  {sp:<6} value wMSE {out[f'warmstart_{sp}']:.4f}")
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = {}
    if which in ("all", "a"):
        check_a()
    if which in ("all", "b"):
        check_b()
    if which in ("all", "c"):
        res["run3_g12"] = check_c("run3", 12, "y_blend", 0.1, 0.01)
        res["run2_g12"] = check_c("run2", 12, "y_blend", 0.1, 0.01)
        dump("step0_production_reference.json", res)
