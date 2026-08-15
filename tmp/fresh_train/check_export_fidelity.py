"""Export fidelity: gen-9 keras (weights.h5) vs gen-9 model.onnx on CPU.
Also: muon vs adam routing per variable, and the empty-board policy/value."""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
REPO = "C:/Users/TryHarder/Desktop/Pythoncode/Grok_Zero"
sys.path.insert(0, REPO)

import numpy as np
import tensorflow as tf

from Connect4.game import Connect4
from Connect4.config import build_config, train_config, optimizer_config
from Connect4.Build_Model import build_model

GEN = 9
RUN = f"{REPO}/Connect4/Grok_Zero_Train/{GEN}"

game = Connect4()
configs = (build_config, train_config, optimizer_config)
model = build_model(game.get_input_state().shape, game.policy_shape, configs)
model.load_weights(f"{RUN}/model.weights.h5")

# ---- inputs: empty board + 100 random-ish boards ----
empty = game.get_input_state().astype(np.float32)[None]  # (1,6,7,4)
print("empty board input: unique per channel:",
      [np.unique(empty[0, :, :, c]).tolist() for c in range(4)])

rng = np.random.default_rng(0)
rand = rng.integers(-1, 2, size=(100, 6, 7, 4)).astype(np.float32)
X = np.concatenate([empty, rand], axis=0)

kp, kv = model.predict(X, batch_size=101, verbose=0)
kp = np.asarray(kp); kv = np.asarray(kv)

import onnxruntime as rt
sess = rt.InferenceSession(f"{RUN}/model.onnx", providers=["CPUExecutionProvider"])
print("onnx inputs:", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
print("onnx outputs:", [(o.name, o.shape, o.type) for o in sess.get_outputs()])
op, ov = sess.run(["policy", "value"], {"inputs": X})

print("\n---- keras vs onnx agreement (gen 9, CPU) ----")
print("max |policy diff| :", np.abs(kp - op).max())
print("max |value diff|  :", np.abs(kv - ov).max())
print("argmax policy agreement:", (kp.argmax(1) == op.argmax(1)).mean())

print("\n---- empty board (gen 9) ----")
np.set_printoptions(precision=5, suppress=True)
print("keras policy:", kp[0])
print("onnx  policy:", op[0])
print("keras argmax:", kp[0].argmax(), " value:", float(kv[0][0]))
print("onnx  argmax:", op[0].argmax(), " value:", float(ov[0][0]))

# ---- muon routing check ----
from Net.Optimizers.muon import Muon
kwargs = dict(optimizer_config["kwargs"])
kwargs["learning_rate"] = 3e-3
opt = Muon(**kwargs)
muon_vars, adam_vars = [], []
for v in model.trainable_variables:
    (adam_vars if opt._should_use_adamw(v) else muon_vars).append((v.path, tuple(v.shape)))
print("\n---- muon routing ----")
print(f"{len(muon_vars)} muon vars, {len(adam_vars)} adam vars")
print("MUON (orthogonalized):")
for p, s in muon_vars: print("   ", p, s)
print("ADAM:")
for p, s in adam_vars: print("   ", p, s)
