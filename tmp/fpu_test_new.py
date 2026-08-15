"""Does FPU make the search find better moves at the PRODUCTION sim count?

Earlier this was tested on run 2's gen-7 net and found to do nothing -- but that
value head was worse than a board-blind baseline, so its Q values sat near zero and
'Q=0' vs 'parentQ - r' were nearly the same number. FPU can only matter when Q is
meaningful. Run 3's head now separates positions, so re-testing is justified.

Method (avoids grading either rule against itself):
  reference = the move both rules agree on at DEEP_SIMS. At that depth the
              initialisation of unvisited children is washed out, and requiring
              agreement discards positions where the answer is genuinely unsettled.
  score     = how often each rule reproduces that move at the production sim count.

Also reports search shape and, crucially, breaks results down by |root Q| -- the
mechanism predicts FPU matters most where the eval is far from zero.
"""
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import onnxruntime as rt

rt.set_default_logger_severity(4)

from MCTS import MCTS
from Connect4.game import Connect4

MODEL = "Connect4/Grok_Zero_Train/6/model.onnx"
PROD_SIMS = 375
DEEP_SIMS = 2000
N_POS = 180
# 0.0 = "an untried move is worth exactly the parent's average" (no pessimism).
# Narrowed to the only arm that beat the baseline, so the sample can be tripled
# for the same compute -- 2 discordant positions out of 53 proved nothing.
REDUCTIONS = [0.0]
# Fixed so the reference set stays IDENTICAL to the previous run (same seed, same
# two reference rules) -- the arms are then a paired comparison on the same positions,
# and re-running 0.25 reproduces its earlier number as a consistency check.
REF_REDUCTION = 0.25

_original = MCTS._get_best_PUCT_score_index


def make_fpu(reduction):
    def puct_fpu(child_prob_priors, child_values, child_visits, parent_visits,
                 c_puct_init, c_puct_base):
        U = child_prob_priors * ((parent_visits ** 0.5) / (child_visits + 1)) * (
            c_puct_init + np.log((parent_visits + c_puct_base + 1) / c_puct_base))
        mask = child_visits > 0
        total = child_visits.sum()
        parent_q = (child_values.sum() / total) if total > 0 else 0.0
        Q = np.full(child_values.shape, parent_q - reduction, dtype=np.float32)
        Q[mask] = child_values[mask] / child_visits[mask]
        return int(np.argmax(Q + U))
    return puct_fpu


def set_rule(fn):
    MCTS._get_best_PUCT_score_index = staticmethod(fn if fn else _original)


def pv_depth(root):
    node, d = root, 0
    while True:
        visits = getattr(node, "child_visits", None)
        kids = getattr(node, "children", None)
        if not kids or visits is None or len(kids) == 0:
            return d
        v = np.asarray(visits, dtype=np.float64)
        if v.max() <= 0 or int(np.argmax(v)) >= len(kids):
            return d
        node = kids[int(np.argmax(v))]
        d += 1


def search(game, session, sims, fn):
    set_rule(fn)
    try:
        t = MCTS(game=game, session=session, use_njit=False, c_puct_init=1.5,
                 use_dirichlet=False, tau=0, fast_find_win=False)
        t.run(iteration_limit=sims, use_bar=False)
        v = np.asarray(t.root.child_visits, dtype=np.float64)
        tot = v.sum()
        if tot <= 0:
            return None
        p = v / tot
        nz = p[p > 0]
        vals = np.asarray(t.root.child_values, dtype=np.float64)
        return dict(best=int(np.argmax(v)),
                    eff_b=float(np.exp(-(nz * np.log(nz)).sum())),
                    root_q=float(vals.sum() / tot),
                    pv=pv_depth(t.root))
    finally:
        set_rule(None)


def collect(session, n):
    rng = np.random.default_rng(0)
    out = []
    while len(out) < n:
        g = Connect4()
        for _ in range(rng.integers(6, 24)):
            la = g.get_legal_actions()
            if len(la) == 0:
                break
            s = g.get_input_state()[None].astype(np.float32)
            pol, _ = session.run(None, {"inputs": s})
            pr = pol[0][np.asarray(la, dtype=np.int64)]
            pr = pr / pr.sum()
            g.do_action(int(rng.choice(np.asarray(la, dtype=np.int64), p=pr)))
            if g.check_win() != -2:
                break
        if g.check_win() == -2 and len(g.get_legal_actions()) > 1:
            snap = Connect4.__new__(Connect4)
            snap.board = g.board.copy()
            snap.next_player = g.next_player
            snap.action_history = list(g.action_history)
            snap.policy_shape = g.policy_shape
            out.append(snap)
    return out


def main():
    sess = rt.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    pos = collect(sess, N_POS)
    print(f"net: run3 gen 6 | production {PROD_SIMS} sims | reference {DEEP_SIMS} sims")
    print(f"{len(pos)} positions from net-guided play\n")

    fpus = {r: make_fpu(r) for r in sorted(set(REDUCTIONS) | {REF_REDUCTION})}
    ref_fn = fpus[REF_REDUCTION]

    kept = []
    for g in pos:
        a = search(g, sess, DEEP_SIMS, None)
        b = search(g, sess, DEEP_SIMS, ref_fn)
        if a and b and a["best"] == b["best"]:
            kept.append((g, a["best"], a["root_q"]))
    print(f"reference settled on {len(kept)}/{len(pos)} positions "
          f"(both rules agree at {DEEP_SIMS} sims)\n")

    arms = [("current (Q=0)", None)] + [
        (("FPU = parent Q" if r == 0 else f"FPU parent-{r}"), fpus[r])
        for r in REDUCTIONS]
    res = {}
    for label, fn in arms:
        rows = []
        for g, ref_best, ref_q in kept:
            r = search(g, sess, PROD_SIMS, fn)
            if r:
                rows.append((r["best"] == ref_best, abs(ref_q), r["eff_b"], r["pv"]))
        res[label] = rows

    print(f"{'rule':<18} {'agrees with deep search':>24} {'eff branch':>12} {'PV depth':>10}")
    for label, _ in arms:
        rows = res[label]
        acc = np.mean([r[0] for r in rows])
        se = np.sqrt(acc * (1 - acc) / max(len(rows), 1))
        print(f"{label:<18} {acc:>17.1%} +/-{se:>4.1%} "
              f"{np.mean([r[2] for r in rows]):>12.2f} "
              f"{np.mean([r[3] for r in rows]):>10.1f}")

    # Paired comparison: the arms ran on the SAME positions, so the honest statistic
    # is the discordant count (McNemar), not two independent proportions.
    if len(arms) == 2:
        a = [r[0] for r in res[arms[0][0]]]
        b = [r[0] for r in res[arms[1][0]]]
        only_a = sum(1 for x, y in zip(a, b) if x and not y)
        only_b = sum(1 for x, y in zip(a, b) if y and not x)
        n_dis = only_a + only_b
        # exact two-sided binomial on the discordant pairs, p = 0.5 under the null
        from math import comb
        k = min(only_a, only_b)
        p = min(1.0, 2 * sum(comb(n_dis, i) for i in range(k + 1)) / (2 ** n_dis)) \
            if n_dis else 1.0
        print(f"\nPAIRED (McNemar) on {len(a)} positions")
        print(f"  only {arms[0][0]} correct : {only_a}")
        print(f"  only {arms[1][0]} correct : {only_b}")
        print(f"  discordant {n_dis}, exact two-sided p = {p:.3f}"
              f"   -> {'SIGNIFICANT' if p < 0.05 else 'not significant'}")

    print(f"\nby |root Q| -- the mechanism predicts FPU matters most when |Q| is LARGE")
    edges = [(0.0, 0.2, "|Q| < 0.2"), (0.2, 0.5, "0.2-0.5"), (0.5, 1.01, "|Q| > 0.5")]
    print(f"{'bucket':<12} {'n':>4} " + "".join(f"{lbl:>20}" for lbl, _ in arms))
    for lo, hi, name in edges:
        line = f"{name:<12}"
        n_shown = None
        cells = ""
        for label, _ in arms:
            rows = [r for r in res[label] if lo <= r[1] < hi]
            n_shown = len(rows)
            cells += f"{(np.mean([r[0] for r in rows]) if rows else float('nan')):>19.1%} "
        print(line + f"{n_shown:>4} " + cells)


if __name__ == "__main__":
    main()
