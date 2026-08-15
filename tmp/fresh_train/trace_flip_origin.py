"""For every generation: the net's empty-board policy vs the empty-board target
of the window it was trained on, and the data its own play then produced.
Shows whether the col3->col1 flip appears first in a NET or in the DATA."""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
REPO = "C:/Users/TryHarder/Desktop/Pythoncode/Grok_Zero"
sys.path.insert(0, REPO)

import numpy as np
import h5py as h5
np.set_printoptions(precision=4, suppress=True)

from Connect4.game import Connect4
from Connect4.config import build_config, train_config, optimizer_config
game = Connect4()
empty = game.get_input_state().astype(np.float32)[None]

def data_mean(gen):
    with h5.File(f"{REPO}/Connect4/Grok_Zero_Train/{gen}/Self_Play_Data.h5", "r") as f:
        nd = (len(f.keys()) - 1) // 3
        pols = [np.asarray(f[f"policies_{d}"][0], dtype=np.float64) for d in range(nd)]
    return np.mean(pols, 0)

def window(gen_trained_on):  # replicate Create_Train_Test_Split file selection
    g = gen_trained_on  # training that produced gen g+1 weights saw folders 0..g
    files = list(range(0, g + 1))[-(4 + 1):]
    if g >= 2:
        files = files[1:]
    return files

import tensorflow as tf
from Connect4.Build_Model import build_model
configs = (build_config, train_config, optimizer_config)
model = build_model(game.get_input_state().shape, game.policy_shape, configs)

dm = {g: data_mean(g) for g in range(10)}

print("net gen | trained on gens | window target (argmax) | net empty-board policy (argmax) | own-play data (argmax)")
for g in range(10):
    model.load_weights(f"{REPO}/Connect4/Grok_Zero_Train/{g}/model.weights.h5")
    p = np.asarray(model.predict(empty, verbose=0)[0])[0]
    if g == 0:
        wtxt, ttxt = "none (init)", "-"
    else:
        w = window(g - 1)
        tgt = np.mean([dm[x] for x in w], 0)
        wtxt, ttxt = str(w), f"c1={tgt[1]:.3f} c3={tgt[3]:.3f} am={tgt.argmax()}"
    own = dm[g]
    print(f"  {g}     | {wtxt:14} | {ttxt:28} | c1={p[1]:.3f} c3={p[3]:.3f} am={p.argmax()} | c1={own[1]:.3f} c3={own[3]:.3f} am={own.argmax()}")
