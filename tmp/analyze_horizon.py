"""Is the value collapse a data limit or a network failure? Bucket by plies-to-end.

For each position we know how many plies remain until the game actually ended. The MCTS
sees ~7 plies. So:

  |target| stays high far from the end, |pred| collapses  -> the network is failing to
                                                              fit a label it was given
  |target| itself collapses far from the end              -> the DATA cannot teach it;
                                                              the label is already ~0

Those need completely different fixes, and the ply-index buckets used so far cannot tell
them apart.

Usage: python tmp/analyze_horizon.py [generation]
"""
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import h5py as h5
import tensorflow as tf

GEN = int(sys.argv[1]) if len(sys.argv) > 1 else 4
MODEL_GEN = int(sys.argv[2]) if len(sys.argv) > 2 else GEN
RUN = "Connect4/Grok_Zero_Train"
BUCKETS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 11), (12, 15), (16, 23), (24, 99)]


def load_games(path):
    """Per game: boards, policies, targets, and plies remaining until the game ended."""
    out = []
    with h5.File(path, "r") as f:
        n = int(f["game_stats"][2])
        aug = ((len(f.keys()) - 1) // 3) // n
        for g in range(n):
            i = g * aug  # first augmentation only, mirrors are the same game
            b = f[f"boards_{i}"][:]
            p = f[f"policies_{i}"][:]
            v = f[f"values_{i}"][:][:, 0]
            out.append((b, p, v, np.arange(len(b))[::-1]))
    return out


def main():
    data_path = f"{RUN}/{GEN}/Self_Play_Data.h5"
    weights = f"{RUN}/{MODEL_GEN}/model.weights.h5"
    if not os.path.exists(weights):
        print(f"no weights at {weights}")
        return

    games = load_games(data_path)
    X = np.concatenate([g[0] for g in games]).astype(np.float32)
    P = np.concatenate([g[1] for g in games]).astype(np.float64)
    T = np.concatenate([g[2] for g in games]).astype(np.float64)
    R = np.concatenate([g[3] for g in games])          # plies remaining
    print(f"gen {GEN}: {len(games)} games, {len(X)} positions, "
          f"mean length {np.mean([len(g[0]) for g in games]):.1f} plies\n")

    from Connect4.Build_Model import build_model
    from Connect4.game import Connect4
    from Connect4.config import build_config, train_config, optimizer_config
    game = Connect4()
    model = build_model(game.get_input_state().shape, game.policy_shape,
                        (build_config, train_config, optimizer_config))
    model.load_weights(weights)
    pp, pv = model.predict(X, batch_size=4096, verbose=0)
    pv = pv[:, 0].astype(np.float64)
    pp = np.clip(pp.astype(np.float64), 1e-12, 1.0)

    print(f"{'plies to end':>14} {'n':>7} {'|target|':>9} {'|pred|':>8} {'shrink':>8} "
          f"{'MSE':>8} {'polKL':>8} {'top1':>7}")
    print("-" * 74)
    for lo, hi in BUCKETS:
        m = (R >= lo) & (R <= hi)
        if m.sum() == 0:
            continue
        t, p, q = T[m], pv[m], P[m]
        ce = float(np.mean(-(q * np.log(pp[m])).sum(1)))
        ent = float(np.mean(-(q * np.log(np.clip(q, 1e-12, 1))).sum(1)))
        top1 = float(np.mean(pp[m].argmax(1) == q.argmax(1)))
        at, ap = np.abs(t).mean(), np.abs(p).mean()
        print(f"{f'{lo}-{hi}':>14} {m.sum():>7} {at:>9.4f} {ap:>8.4f} "
              f"{ap/at if at else float('nan'):>8.2f} {np.mean((p-t)**2):>8.4f} "
              f"{ce-ent:>8.4f} {top1:>6.1%}")

    # Signed toward the truth, so wins and losses reinforce instead of cancelling.
    dec = np.abs(T) > 0.5
    conf = pv * np.sign(T)
    print(f"\ndecisive targets (|t|>0.5): {dec.mean():.1%} of positions")
    print(f"{'plies to end':>14} {'n':>7} {'|t|':>7} {'conf':>7} {'sign ok':>8} "
          f"{'blunder':>8}")
    print("-" * 52)
    for lo, hi in BUCKETS:
        m = (R >= lo) & (R <= hi) & dec
        if m.sum() < 20:
            continue
        print(f"{f'{lo}-{hi}':>14} {m.sum():>7} {np.abs(T[m]).mean():>7.3f} "
              f"{conf[m].mean():>7.3f} {np.mean(conf[m] > 0):>7.1%} "
              f"{np.mean(conf[m] < -0.3):>7.1%}")
    print("  conf = predicted value signed toward the true outcome (higher = more right)")
    print("  blunder = confidently wrong, conf < -0.3")


if __name__ == "__main__":
    main()
