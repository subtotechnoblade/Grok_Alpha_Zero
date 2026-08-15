"""Ground-truth value test. No solver binary needed for the openings: Connect4 is
solved and the perfect-play result of every first move is published.

  col 3 (centre)  -> first player WINS      (+1)
  col 2, 4        -> DRAW                   ( 0)
  col 0,1,5,6     -> first player LOSES     (-1)

Plus two tactical ground truths generated from random positions, where the answer
is certain regardless of any theory:
  A. mover has an immediate 4-in-a-row available     -> true value +1
  B. mover has no win, opponent has >=2 winning cols -> true value -1 (can only block one)

Sign convention: the stored value target at a ply is from the perspective of the
player to move there, so the net's output for a 1-stone board is p+1's view and
the opener's value is its negation.

Read-only, CPU only.
"""
import os
import sys
import glob
import copy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import onnxruntime as rt

rt.set_default_logger_severity(4)

from Connect4.game import Connect4

TRUTH = np.array([-1.0, -1.0, 0.0, 1.0, 0.0, -1.0, -1.0])
GENS = sorted((int(os.path.basename(p)) for p in glob.glob("Connect4/Grok_Zero_Train/*")
               if os.path.exists(os.path.join(p, "model.onnx"))))


def val_of(sess, game):
    s = game.get_input_state()[None].astype(np.float32)
    pol, val = sess.run(None, {"inputs": s})
    return float(val[0][0]), pol[0]


def winning_cols(game, player):
    out = []
    for c in game.get_legal_actions():
        g = copy.deepcopy(game)
        g.do_action(int(c))
        if g.check_win() == player:
            out.append(int(c))
    return out


def make_tactical(n_target=300, seed=0):
    """Random positions tagged with a CERTAIN true value."""
    rng = np.random.default_rng(seed)
    wins, losses = [], []
    tries = 0
    while (len(wins) < n_target or len(losses) < n_target) and tries < 40000:
        tries += 1
        g = Connect4()
        depth = int(rng.integers(4, 30))
        ok = True
        for _ in range(depth):
            la = g.get_legal_actions()
            if len(la) == 0:
                ok = False
                break
            g.do_action(int(rng.choice(np.asarray(la, dtype=np.int64))))
            if g.check_win() != -2:
                ok = False
                break
        if not ok:
            continue
        mover = g.next_player
        my_wins = winning_cols(g, mover)
        if my_wins:
            if len(wins) < n_target:
                wins.append(copy.deepcopy(g))
            continue
        opp_wins = winning_cols(g, -mover)
        if len(opp_wins) >= 2 and len(losses) < n_target:
            losses.append(copy.deepcopy(g))
    return wins, losses


print("=" * 78)
print("A. OPENING BOOK vs PUBLISHED PERFECT PLAY (opener's perspective)")
print("   truth:  col0 -1  col1 -1  col2  0  col3 +1  col4  0  col5 -1  col6 -1")
print(f"{'gen':>4}  " + "  ".join(f"c{c}" for c in range(7)) +
      f"  {'signOK':>7} {'MSE':>7} {'best':>5}")
for gen in GENS:
    m = f"Connect4/Grok_Zero_Train/{gen}/model.onnx"
    try:
        sess = rt.InferenceSession(m, providers=["CPUExecutionProvider"])
    except Exception:
        continue
    q = np.zeros(7)
    for c in range(7):
        g = Connect4()
        g.do_action(c)
        v, _ = val_of(sess, g)
        q[c] = -v                       # negate: state's mover is p+1
    decided = TRUTH != 0
    sign_ok = int(np.sum(np.sign(q[decided]) == np.sign(TRUTH[decided])))
    mse = float(np.mean((q - TRUTH) ** 2))
    print(f"{gen:>4}  " + "  ".join(f"{x:+.2f}" for x in q) +
          f"  {sign_ok:>2}/5   {mse:>7.3f} {int(np.argmax(q)):>5}")

print()
print("=" * 78)
print("B. TACTICAL GROUND TRUTH (certain answers, no theory needed)")
wins, losses = make_tactical()
print(f"   {len(wins)} positions with an immediate win (true +1), "
      f"{len(losses)} lost-to-double-threat (true -1)")
print(f"{'gen':>4} {'win: mean V':>12} {'V>0':>7} {'V>0.5':>7} | "
      f"{'loss: mean V':>13} {'V<0':>7} {'V<-0.5':>8} | {'MSE':>7}")
for gen in GENS:
    m = f"Connect4/Grok_Zero_Train/{gen}/model.onnx"
    try:
        sess = rt.InferenceSession(m, providers=["CPUExecutionProvider"])
    except Exception:
        continue
    wv = np.array([val_of(sess, g)[0] for g in wins])
    lv = np.array([val_of(sess, g)[0] for g in losses])
    mse = float((np.sum((wv - 1) ** 2) + np.sum((lv + 1) ** 2)) / (len(wv) + len(lv)))
    print(f"{gen:>4} {wv.mean():>+12.3f} {np.mean(wv > 0):>7.1%} "
          f"{np.mean(wv > 0.5):>7.1%} | {lv.mean():>+13.3f} {np.mean(lv < 0):>7.1%} "
          f"{np.mean(lv < -0.5):>8.1%} | {mse:>7.3f}")
