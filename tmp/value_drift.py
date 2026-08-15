"""Evaluate every generation's model on ONE fixed position set to isolate model drift.

Comparing each model against its own generation's games confounds model change with data
change. Here the probe set is frozen (generation 1, still diverse openings), so any
movement is the network.

v(empty)  = value of the empty board, player -1 (first) to move. If the net is genuinely
            learning the first-player win this rises toward +1. If it rises far past what
            the net can justify, that is the echo chamber.
v(centre) = value after 1.centre, player +1 (second) to move; should mirror negative.
"""
import os
import sys
from glob import glob
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import h5py as h5
import tensorflow as tf
RUN = "Connect4/Grok_Zero_Train"
PROBE_GEN = 1


def load_probe():
    with h5.File(f"{RUN}/{PROBE_GEN}/Self_Play_Data.h5", "r") as f:
        n = int(f["game_stats"][2])
        aug = ((len(f.keys()) - 1) // 3) // n
        X, empty, centre = [], None, None
        for g in range(n):
            b = f[f"boards_{g * aug}"][:]
            X.append(b)
            if empty is None:
                empty = b[0]
            if centre is None and len(b) > 1:
                occ = (b[1][..., -1] != 0) & ~(b[0][..., -1] != 0)
                if len(np.argwhere(occ)) and int(np.argwhere(occ)[0][1]) == 3:
                    centre = b[1]
    return np.concatenate(X).astype(np.float32), empty[None].astype(np.float32), \
        centre[None].astype(np.float32)


def main():
    X, empty, centre = load_probe()
    print(f"probe: {len(X)} positions from gen {PROBE_GEN}\n")

    from Connect4.Build_Model import build_model
    from Connect4.game import Connect4
    from Connect4.config import build_config, train_config, optimizer_config
    game = Connect4()
    model = build_model(game.get_input_state().shape, game.policy_shape,
                        (build_config, train_config, optimizer_config))

    print(f"{'model':>6} {'v(empty)':>9} {'v(centre)':>10} {'mean v':>8} {'mean|v|':>8} "
          f"{'p(c3)':>7} {'H(pol)':>7}")
    print("-" * 62)
    for d in sorted(glob(f"{RUN}/*/"), key=lambda p: int(Path(p).name)):
        w = os.path.join(d, "model.weights.h5")
        if not os.path.exists(w):
            continue
        model.load_weights(w)
        pe = model.predict(empty, verbose=0)
        pc = model.predict(centre, verbose=0)
        pp, pv = model.predict(X, batch_size=4096, verbose=0)
        pol = np.clip(pe[0][0].astype(np.float64), 1e-12, 1)
        print(f"{int(Path(d).name):>6} {pe[1][0, 0]:>+9.4f} {pc[1][0, 0]:>+10.4f} "
              f"{pv[:, 0].mean():>+8.4f} {np.abs(pv[:, 0]).mean():>8.4f} "
              f"{pol[3]:>7.1%} {-(pol * np.log(pol)).sum():>7.4f}")


if __name__ == "__main__":
    main()
