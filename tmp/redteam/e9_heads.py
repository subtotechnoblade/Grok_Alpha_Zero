"""E9: is the trunk missing the tactic, or is only the VALUE head missing it?

Same random-play win-in-1 positions, both heads:
  policy -> does argmax (over legal moves) land on a winning column?
  value  -> is the output near +1?

If the policy head finds the win but the value head says "even", the shared trunk
DOES represent the tactic and the failure is specific to the value head. That is a
completely different repair from "the network cannot see it".

Random-legal baseline for policy top-1 is ~1/n_legal (~15-20%).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tmp", "learnability"))
sys.path.insert(0, HERE)
os.chdir(REPO)

import numpy as np
import data as D
from e7_settle import random_win1, win1_from_games, win_cols, landing

ROOT = D.ROOTS["run2"]
COLS = 7


def sess(path):
    import onnxruntime as rt
    rt.set_default_logger_severity(4)
    return rt.InferenceSession(path, providers=["CPUExecutionProvider"])


def infer(s, X):
    P, V = [], []
    for i in range(0, len(X), 2048):
        p, v = s.run(None, {"inputs": np.asarray(X[i:i + 2048], np.float32)})
        P.append(p)
        V.append(v[:, 0])
    return np.concatenate(P), np.concatenate(V)


def score(s, X):
    P, V = infer(s, X)
    top1, mass, nlegal = [], [], []
    for i, st in enumerate(X):
        board = st[..., 3]
        mover = -int(st[..., 0].ravel()[0])
        wins = win_cols(board, mover)
        legal = np.array([landing(board, c) >= 0 for c in range(COLS)])
        p = P[i] * legal
        ssum = p.sum()
        p = p / ssum if ssum > 0 else p
        top1.append(int(np.argmax(p)) in wins)
        mass.append(p[wins].sum())
        nlegal.append(int(legal.sum()))
    return (np.mean(top1), np.mean(mass), float(np.mean((V - 1) ** 2)),
            np.mean(V > 0), V.mean(), np.mean(1.0 / np.array(nlegal)))


def main():
    sets = {
        "gen12 self-play (in-window)": win1_from_games(D._read_gen(ROOT, 12)),
        "gen13 self-play (next gen)": win1_from_games(D._read_gen(ROOT, 13)),
        "RANDOM play seed0": random_win1(1200, seed=0),
        "RANDOM play seed1": random_win1(1200, seed=1),
    }
    print("WIN-IN-1 positions. Policy: can it find the winning move? "
          "Value: does it know it is won?")
    print("=" * 104)
    hdr = (f"{'model':<7} {'set':<30} {'n':>5} | {'POLICY top1':>12} "
           f"{'P(win col)':>11} {'chance':>7} | {'VALUE mean':>11} "
           f"{'MSE vs +1':>10} {'sign ok':>8}")
    print(hdr)
    print("-" * len(hdr))
    for M in (13, 14, 15):
        p = f"{ROOT}/{M}/model.onnx"
        if not os.path.exists(p):
            continue
        s = sess(p)
        for name, X in sets.items():
            if len(X) == 0:
                continue
            t1, mass, mse, sgn, mv, ch = score(s, X)
            print(f"m{M:<6} {name:<30} {len(X):>5} | {t1*100:>11.1f}% "
                  f"{mass*100:>10.1f}% {ch*100:>6.1f}% | {mv:>+11.3f} "
                  f"{mse:>10.3f} {sgn*100:>7.1f}%", flush=True)
        print()


if __name__ == "__main__":
    main()
