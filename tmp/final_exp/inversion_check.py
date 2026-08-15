"""How much did tmp/ab_heads.py's inverted mover distort the win-in-1 diagnostic?

ab_heads.py reads +1 as the first mover; the stored states make -1 the first
mover (chan0 = -next_player). The rule is therefore correct on ODD plies and
inverted on EVEN plies. On an inverted position the "mover" is the side that is
NOT to move, so a "win-in-1" is really the OPPONENT's threat and the true value
for the side to move is near -1, not +1.

Run the same archived models against both collections.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
os.chdir(ROOT)

import numpy as np
from gtlib import collect_ground_truth, run_model, session

CASES = [("run2", "tmp/run2_archive_gen0-19_oldvaluehead", [6, 12, 18]),
         ("run3", "tmp/run3_archive_gen0-15_bottleneck_preweightfix", [6, 12, 15])]

print(f"{'run':>5} {'gen':>4} {'convention':>12} {'n':>6} {'mean V':>9} "
      f"{'V>0':>7} {'V>0.9':>7} {'MSE vs +1':>10} {'blk top1':>9}")
for name, root, gens in CASES:
    for g in gens:
        data = f"{root}/{g}/Self_Play_Data.h5"
        onnx = f"{root}/{g}/model.onnx"
        if not (os.path.exists(data) and os.path.exists(onnx)):
            continue
        sess = session(onnx)
        for label, inv in (("correct", False), ("ab_heads", True)):
            w, b = collect_ground_truth(data, stride=4, inverted=inv)
            if not w["X"]:
                continue
            _, V = run_model(sess, np.asarray(w["X"], dtype=np.float32))
            if b["X"]:
                Pb, _ = run_model(sess, np.asarray(b["X"], dtype=np.float32))
                legal = np.asarray(b["mask"])
                P = Pb * legal
                s = P.sum(1, keepdims=True)
                P = np.divide(P, s, out=np.full_like(P, 1 / 7), where=s > 0)
                btop = float(np.mean(P.argmax(1) == np.asarray(b["col"])))
            else:
                btop = float("nan")
            print(f"{name:>5} {g:>4} {label:>12} {len(V):>6} {V.mean():>+9.3f} "
                  f"{np.mean(V > 0):>7.1%} {np.mean(V > 0.9):>7.1%} "
                  f"{np.mean((V - 1) ** 2):>10.3f} {btop:>9.1%}")
        print()
