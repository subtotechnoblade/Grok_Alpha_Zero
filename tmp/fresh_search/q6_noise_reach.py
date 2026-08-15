"""Q6: how much does dirichlet(alpha=1.33, eps=0.25) actually move the ply-0 prior?

Uses the real gen-9 net prior at the empty board, applies the exact mixing formula
from MCTS._apply_dirichlet many times, and reports per-column spread and how often
the noised prior's argmax differs / how often col 3 beats col 1.
"""
import sys
import numpy as np
import onnxruntime as rt

sys.path.insert(0, ".")
from Connect4.game import Connect4

so = rt.SessionOptions()
so.intra_op_num_threads = 2
so.log_severity_level = 3
sess = rt.InferenceSession("Connect4/Grok_Zero_Train/9/model.onnx",
                           sess_options=so, providers=["CPUExecutionProvider"])
game = Connect4()
inp = game.get_input_state()[None].astype(np.float32)
policy, value = sess.run(["policy", "value"], {"inputs": inp})
p = policy[0].astype(np.float64)
p = p / p.sum()
print(f"net ply-0 prior : {np.array2string(p, precision=4, suppress_small=True)}")
print(f"net ply-0 value : {value[0][0]:+.4f}")

alpha, eps = 1.33, 0.25
N = 100_000
noise = np.random.dirichlet(alpha * np.ones(7), size=N)
mixed = (1 - eps) * p + eps * noise
print(f"noised prior std per column: {np.array2string(mixed.std(0), precision=4)}")
print(f"P(argmax(noised prior) per column): "
      f"{np.array2string(np.bincount(np.argmax(mixed, 1), minlength=7) / N, precision=4)}")
print(f"P(noised prior of col3 > col1): {np.mean(mixed[:, 3] > mixed[:, 1]):.4f}")
print(f"P(noised prior of col5 > col1): {np.mean(mixed[:, 5] > mixed[:, 1]):.4f}")
