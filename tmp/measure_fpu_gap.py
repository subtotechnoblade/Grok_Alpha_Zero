"""Quantify the Q=0 search asymmetry, and how much FPU closes it.

The claim: when the mover is LOSING (root value < 0), unvisited children score 0
and therefore outrank every explored move, so the search spreads instead of
deepening. When the mover is WINNING it exploits normally.

So positions must be grouped by the NET'S VALUE, not by side to move (random
midgame positions are favourable for whoever moves, which masks the effect).
Positions come from net-guided play, i.e. the distribution the search really
faces.

Metrics at the root:
  eff branch = exp(entropy of visit distribution)  - how broad
  PV depth   = plies along the most-visited path   - how deep it actually read
  top1       = visit share of the best move        - how committed
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

SIMS = 375
MODEL = "Connect4/Grok_Zero_Train/7/model.onnx"
FPU_REDUCTION = 0.25
_original = MCTS._get_best_PUCT_score_index


def puct_fpu(child_prob_priors, child_values, child_visits, parent_visits,
             c_puct_init, c_puct_base):
    U = child_prob_priors * ((parent_visits ** 0.5) / (child_visits + 1)) * (
        c_puct_init + np.log((parent_visits + c_puct_base + 1) / c_puct_base))
    mask = child_visits > 0
    total = child_visits.sum()
    parent_q = (child_values.sum() / total) if total > 0 else 0.0
    Q = np.full(child_values.shape, parent_q - FPU_REDUCTION, dtype=np.float32)
    Q[mask] = child_values[mask] / child_visits[mask]
    return int(np.argmax(Q + U))


def set_rule(use_fpu):
    MCTS._get_best_PUCT_score_index = staticmethod(puct_fpu if use_fpu
                                                   else _original)


def pv_depth(root):
    """Plies along the most-visited path - how deep the search actually read."""
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


def search(game, session, use_fpu):
    set_rule(use_fpu)
    try:
        tree = MCTS(game=game, session=session, use_njit=False,
                    c_puct_init=1.5, use_dirichlet=False, tau=0,
                    fast_find_win=False)
        tree.run(iteration_limit=SIMS, use_bar=False)
        v = np.asarray(tree.root.child_visits, dtype=np.float64)
        total = v.sum()
        if total <= 0:
            return None
        p = v / total
        nz = p[p > 0]
        vals = np.asarray(tree.root.child_values, dtype=np.float64)
        return dict(eff_b=float(np.exp(-(nz * np.log(nz)).sum())),
                    top1=float(p.max()),
                    touched=int((v > 0).sum()),
                    root_q=float(vals.sum() / total),
                    pv=pv_depth(tree.root))
    finally:
        set_rule(False)


def collect(session, n_games=10):
    """Positions from net-guided play, tagged with the net's value."""
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n_games):
        g = Connect4()
        for ply in range(14):
            state = g.get_input_state()[None].astype(np.float32)
            pol, val = session.run(None, {"inputs": state})
            out.append((Connect4.__new__(Connect4), float(val[0][0])))
            # deep-copy the game state into the stored object
            gg = out[-1][0]
            gg.board = g.board.copy()
            gg.next_player = g.next_player
            gg.action_history = list(g.action_history)
            gg.policy_shape = g.policy_shape
            la = g.get_legal_actions()
            if len(la) == 0:
                break
            # sample from the prior so games diverge, like tau=1 self play
            pr = pol[0][np.asarray(la, dtype=np.int64)]
            pr = pr / pr.sum()
            g.do_action(int(rng.choice(np.asarray(la, dtype=np.int64), p=pr)))
            if g.check_win() != -2:
                break
    return out


def main():
    sess = rt.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    pos = collect(sess)
    losing = [g for g, v in pos if v < -0.15]
    winning = [g for g, v in pos if v > 0.15]
    print(f"net: gen 7 | {SIMS} sims | FPU = parent Q - {FPU_REDUCTION}")
    print(f"positions: {len(losing)} where the mover is LOSING (net V < -0.15), "
          f"{len(winning)} WINNING (V > +0.15)\n")

    rows = {}
    for use_fpu in (False, True):
        for tag, group in (("losing", losing), ("winning", winning)):
            s = [x for x in (search(g, sess, use_fpu) for g in group) if x]
            rows[(use_fpu, tag)] = {k: float(np.mean([d[k] for d in s]))
                                    for k in s[0]}

    lab = {False: "current (Q=0)", True: f"FPU (parent-{FPU_REDUCTION})"}
    print(f"{'rule':<22} {'mover':>8} {'root Q':>8} {'eff branch':>11} "
          f"{'top1':>7} {'children hit':>13} {'PV depth':>9}")
    for use_fpu in (False, True):
        for tag in ("losing", "winning"):
            r = rows[(use_fpu, tag)]
            print(f"{lab[use_fpu]:<22} {tag:>8} {r['root_q']:>8.3f} "
                  f"{r['eff_b']:>11.2f} {r['top1']:>6.1%} {r['touched']:>13.1f} "
                  f"{r['pv']:>9.1f}")

    print()
    for use_fpu in (False, True):
        lo, wi = rows[(use_fpu, "losing")], rows[(use_fpu, "winning")]
        print(f"{lab[use_fpu]:<22} losing vs winning: "
              f"breadth {lo['eff_b'] / wi['eff_b']:.2f}x, "
              f"PV depth {lo['pv'] - wi['pv']:+.1f} plies, "
              f"top1 {lo['top1'] - wi['top1']:+.1%}")


if __name__ == "__main__":
    main()
