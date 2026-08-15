"""End-to-end Windows smoke test: does run_self_play survive `spawn`?

Runs a tiny real generation (few games, 2 workers) through the actual Self_Play
machinery - shared memory, inference server, h5 writes - in a scratch folder.
Static analysis cannot prove spawn-safety; this can.

Run with:
  C:\\Users\\TryHarder\\anaconda3\\envs\\tensorflow_updated\\python.exe tmp/windows_spawn_smoke.py [--server] [--gpu]
"""
import os
import shutil
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    os.chdir(ROOT)
    import numpy as np
    import h5py as h5
    from Connect4.Connect4 import Connect4, build_config, train_config, optimizer_config
    from Self_Play import run_self_play

    # forward slashes on purpose: Self_Play.py:290 does folder_path.split("/")[-1],
    # so a backslash path (os.path.join on Windows) raises ValueError there.
    # main.py builds "Grok_Zero_Train/{gen}" the same way, ending in the generation number.
    SCRATCH = ROOT.replace("\\", "/") + "/tmp/spawn_smoke/1"
    if os.path.exists(SCRATCH):
        shutil.rmtree(SCRATCH, ignore_errors=True)
    os.makedirs(SCRATCH, exist_ok=True)

    # export the model so generation 1 has something to load
    model_path = os.path.join(SCRATCH, "model.onnx")
    src = os.path.join(ROOT, "tmp", "dml_bench", "model.onnx")
    if os.path.exists(src):
        shutil.copy(src, model_path)
    else:
        import tensorflow as tf
        from Connect4.Build_Model import build_model
        from To_Onnx import convert_to_onnx
        g = Connect4()
        m = build_model(g.get_input_state().shape, g.policy_shape,
                        (build_config, train_config, optimizer_config))
        convert_to_onnx(m, [tf.TensorSpec((None, *g.get_input_state().shape),
                                          tf.float32, name="inputs")], model_path)

    # tiny, safe overrides. NOTE max_actions: the live config has None, which makes
    # Self_Play.play()'s `actions_count < max_actions` raise TypeError immediately.
    tc = dict(train_config)
    tc["max_actions"] = 42
    tc["games_per_generation"] = 6
    tc["num_workers"] = 2
    tc["MCTS_iteration_limit"] = 40
    tc["use_inference_server"] = "--server" in sys.argv
    tc["use_gpu"] = "--gpu" in sys.argv
    tc["use_tensorrt"] = False

    with h5.File(os.path.join(SCRATCH, "Self_Play_Data.h5"), "w", libver="latest") as f:
        f.create_dataset("game_stats", data=np.zeros(6, dtype=np.int64), maxshape=(6,))

    print(f"start method   : {__import__('multiprocessing').get_start_method()}")
    print(f"use_gpu        : {tc['use_gpu']}")
    print(f"inference server: {tc['use_inference_server']}")
    print(f"workers        : {tc['num_workers']}, games: {tc['games_per_generation']}, "
          f"sims: {tc['MCTS_iteration_limit']}")
    print("-" * 70)

    t0 = time.perf_counter()
    run_self_play(Connect4, (build_config, tc, optimizer_config), SCRATCH, 1)
    dt = time.perf_counter() - t0

    with h5.File(os.path.join(SCRATCH, "Self_Play_Data.h5"), "r") as f:
        stats = f["game_stats"][:]
        n_datasets = (len(f.keys()) - 1) // 3
    print("-" * 70)
    print(f"wall {dt:.1f}s")
    print(f"game_stats = {stats.tolist()}")
    print(f"  max game length {stats[0]}, positions {stats[1]}, games {stats[2]}")
    print(f"  -1 wins {stats[3]}, draws {stats[4]}, +1 wins {stats[5]}")
    print(f"  dataset triples: {n_datasets} (expect games x augmentations = "
          f"{tc['games_per_generation']} x 2 = {tc['games_per_generation'] * 2})")
    ok = int(stats[2]) == tc["games_per_generation"] and n_datasets == tc["games_per_generation"] * 2
    print(f"\n{'PASS - spawn path works end to end' if ok else 'FAIL - game count mismatch'}")
    return 0 if ok else 1


if __name__ == "__main__":
    # the guard matters: under spawn every child re-imports this module
    sys.exit(main())
