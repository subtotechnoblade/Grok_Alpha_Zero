"""End-to-end self play through the batched server on the CPU provider (correctness, not speed).
python -m tests.smoke_self_play <scratch_dir> <model.onnx> [leaves_per_batch]"""
import os
import shutil
import sys

import h5py as h5

import Client_Server
from tests.mcts_scenarios import ROOT
from pipeline import Make_Dataset_File


def two_thread_cpu_engine(file_path, providers, max_batch, buckets=None, batch_axis=0):
    # the CPU provider's default pool takes every core and starves the MCTS workers
    import onnxruntime as rt
    options = rt.SessionOptions()
    options.intra_op_num_threads = 2
    return rt.InferenceSession(file_path, sess_options=options, providers=["CPUExecutionProvider"])


Client_Server.make_engine = two_thread_cpu_engine
from Self_Play import run_self_play
from Gomoku.game import Gomoku
from Gomoku.config import build_config, train_config, optimizer_config

if __name__ == "__main__":
    scratch, model = sys.argv[1], sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    folder = os.path.join(scratch, "Grok_Zero_Train", "1")
    shutil.rmtree(os.path.join(scratch, "Grok_Zero_Train"), ignore_errors=True)
    os.makedirs(folder)
    shutil.copy(model, os.path.join(folder, "model.onnx"))
    Make_Dataset_File(folder)
    cfg = {key: train_config[key] for key in train_config.keys()}
    cfg.update(use_gpu=False, use_inference_server=True, num_workers=3, leaves_per_batch=k,
               MCTS_iteration_limit=64, positions_per_generation=120, initial_positions_per_generation=120)
    run_self_play(Gomoku, (build_config, cfg, optimizer_config), folder)
    with h5.File(os.path.join(folder, "Self_Play_Data.h5"), "r") as f:
        stats = f["game_stats"][:]
        n_sets = (len(f.keys()) - 1) // 3
        positions = sum(f[f"boards_{i}"].shape[0] for i in range(0, n_sets, 8))
    ok = n_sets == stats[2] * 8 and positions == stats[1] and stats[3] + stats[4] + stats[5] == stats[2]
    print(f"SMOKE games {stats[2]} positions {stats[1]} (from data {positions}) -> {'CONSISTENT' if ok else 'BROKEN'}")
