"""Single training entry point for every game.

    python pipeline.py --game Connect4
    python pipeline.py --game Gomoku

Run it from the repo root. That puts the root on sys.path, so both the shared
modules (Self_Play, MCTS, ...) and the game packages (Connect4.game, ...) resolve
by exactly one name no matter who launches. The old per game main.py relied on
PyCharm injecting the content root and could not be run from a plain terminal.

Each game keeps its own files together:

    Connect4/
        __init__.py
        game.py            rules + njit kernels
        config.py          hyperparameters
        Build_Model.py
        play.py
        Grok_Zero_Train/   this run's output

This replaces Connect4/main.py, Gomoku/main.py and TicTacToe/main.py, which were
three copies of the same ~360 lines that had already drifted apart.
"""
import os
import shutil
import argparse
import importlib
from pathlib import Path

import h5py as h5
import numpy as np
from glob import glob
import warnings

import multiprocessing as mp

from Game_Tester import Game_Tester

from Self_Play import run_self_play
# would create a folder for the new generation
from Build_Tensorrt import cache_tensorrt
from Compute_Speed import compute_speed

REPO_ROOT = Path(__file__).resolve().parent
RUN_FOLDER_NAME = "Grok_Zero_Train"

# Used when --game is not passed, so hitting Run in an IDE works without arguments.
# Change this to switch which game a bare `python pipeline.py` trains.
DEFAULT_GAME = "Connect4"


class _LazyBuildModel:
    """Defers `import tensorflow` until a child process actually builds a model.

    <game>/Build_Model.py imports tensorflow at module scope. Importing it in the
    parent is merely wasteful under spawn, but under fork (linux / WSL) every self
    play worker inherits those pages: measured on WSL at 10 workers the parent hit
    7.9 GB anon-rss and the OOM killer took it mid generation.

    Only the three functions that build a model ever call this, and all of them
    already run inside their own mp.Process. Module level class with a plain string
    attribute, so it still pickles for spawn.
    """

    def __init__(self, game):
        self.game = game

    def __call__(self, *args, **kwargs):
        module = importlib.import_module(f"{self.game}.Build_Model")
        return module.build_model(*args, **kwargs)


def load_game(name):
    """Import one game package and hand back everything the pipeline needs.

    game.py and config.py are cheap and load eagerly. Build_Model is wrapped in
    _LazyBuildModel so tensorflow never enters this process.
    """
    game_dir = REPO_ROOT / name
    if not (game_dir / "game.py").exists():
        available = sorted(p.parent.name for p in REPO_ROOT.glob("*/game.py"))
        raise SystemExit(f"No game named {name!r}. Available: {', '.join(available)}")

    game_module = importlib.import_module(f"{name}.game")
    config_module = importlib.import_module(f"{name}.config")

    try:
        game_class = getattr(game_module, name)
    except AttributeError:
        raise SystemExit(f"{name}/game.py defines no class called {name!r}")

    missing = [c for c in ("build_config", "train_config", "optimizer_config")
               if not hasattr(config_module, c)]
    if missing:
        raise SystemExit(
            f"{name}/config.py is on the old config schema, missing: {', '.join(missing)}.\n"
            f"Connect4/config.py is the current shape: optimizer settings live in\n"
            f"optimizer_config with an lr_schedule function, not inline in train_config.")

    configs = (config_module.build_config,
               config_module.train_config,
               config_module.optimizer_config)
    return game_class, _LazyBuildModel(name), configs, game_dir / RUN_FOLDER_NAME


def gen_path(run_dir, generation):
    """One generation's folder as a forward slash string.

    Absolute, so it no longer depends on the working directory, and posix style so
    the downstream f-string joins and any remaining split("/") stay correct.
    """
    return (Path(run_dir) / str(generation)).as_posix()


def safe_rmtree(path, run_dir):
    """rmtree that refuses to touch anything outside this run's folder.

    run_dir now lives inside the source tree, so a path bug here would delete code
    rather than scratch files.
    """
    path, run_dir = Path(path).resolve(), Path(run_dir).resolve()
    if run_dir.name != RUN_FOLDER_NAME:
        raise RuntimeError(f"run_dir must end in {RUN_FOLDER_NAME}, got {run_dir}")
    if path != run_dir and run_dir not in path.parents:
        raise RuntimeError(f"refusing to delete {path}, it is outside {run_dir}")
    if path.exists():
        shutil.rmtree(path)


def Validate_Configs(configs):
    build_config, train_config, optimizer_config = configs
    import onnxruntime as rt
    if train_config["total_generations"] <= 0:
        raise ValueError("Total generations can't less than or equal to zero")

    if train_config["games_per_generation"] < 2:
        raise ValueError("Games per generation can't be less than 2")

    if not train_config["use_gpu"] and train_config["use_tensorrt"]:
        raise ValueError("You must use the GPU for tensorrt")

    available_providers = rt.get_available_providers()
    if train_config["use_tensorrt"] and "TensorrtExecutionProvider" not in available_providers:
        raise RuntimeError("Please install tensorrt as onnxruntime doesn't detect TensorrtExecutionProvider")

    if train_config["use_gpu"] and not train_config[
        "use_tensorrt"] and ("CUDAExecutionProvider" not in available_providers) and "DmlExecutionProvider" not in available_providers:
        raise RuntimeError("Please install CUDA as onnxruntime doesn't detect CUDAExecutionProvider")

    if train_config["use_gumbel"]:
        if not train_config.get("m", False):
            raise ValueError("Parameter m is missing in train config")

        if not train_config.get("c_visit", False):
            raise ValueError("Parameter c_visit is missing in train config")

        if not train_config.get("c_scale", False):
            raise ValueError("Parameter c_scale is missing in train config")

        if train_config.get("time_limit") is not None:
            raise ValueError("time_limit must be None for gumbel alphazero as gumbel uses iteration_limit")

        m = train_config.get("m")
        s = bin(m).count("1")
        min_iterations = 2 * m - s - (s == 1)
        # fancy way of doing f(128) = 128 + 64 + 32 + 16 + 8 + 4 + 2
        if train_config["MCTS_iteration_limit"] < min_iterations:
            raise ValueError(f"At minimum there needs to be {min_iterations + 1} MCTS iterations for sequential halving to be effective.")

    mixed_precision_policy = build_config.get("mixed_precision", None)
    if mixed_precision_policy is not None:
        if mixed_precision_policy == "mixed_bfloat16":
            raise ValueError("mixed_bfloat16 isn't supported, its not because of me, blame tf2onnx for not supporting it, I have no workaround")
        elif mixed_precision_policy != "mixed_float16":
            raise ValueError(
                f"mixed_precision param is invalid got: {mixed_precision_policy}, should be None, mixed_float16, mixed_bfloat16")
    if mixed_precision_policy == "mixed_bfloat16" and not train_config["use_gpu"]:
        warnings.warn("Using float16 as the compute type for CPU is extremely slow!")
    if optimizer_config["optimizer"].lower() not in ["adam", "nadam", "muon"]:
        raise ValueError(f"Optimizer must be either adam, nadam, or muon got {optimizer_config['optimizer']}.")
    try:
        learning_rate = optimizer_config["kwargs"]["learning_rate"](0)
    except TypeError:
        learning_rate = optimizer_config["kwargs"]["learning_rate"]
    if not (isinstance(learning_rate, int) or isinstance(learning_rate, float)):
        raise ValueError("learning rate must be an int or float, if it is a schedule it must return an int or float")


def Make_Generation_Folder(run_dir, generation):
    os.makedirs(gen_path(run_dir, generation), exist_ok=False)


def Make_Dataset_File(folder_path):
    with h5.File(f"{folder_path}/Self_Play_Data.h5", "w", libver="latest") as file:
        file.create_dataset(f"game_stats", maxshape=(6,), dtype=np.uint32, data=np.zeros(6, ))
        # max_actions, total_actions, num_unaugmented_games, player -1 wins, draws, player 1 wins


def Print_Stats(folder_path):
    with h5.File(f"{folder_path}/Self_Play_Data.h5", "r", libver="latest") as file:
        max_actions, total_actions, num_unaugmented_games, player1_wins, draws, player2_wins = file["game_stats"][:]
        print("---------Game Statistics---------")
        print(f"Longest game is: {max_actions} actions long!")
        print(f"Average moves: {round(total_actions / num_unaugmented_games, 4)}")
        print(f"Player -1 winrate: {round(player1_wins / num_unaugmented_games, 4)}")
        print(f"Draw rate: {round(draws / num_unaugmented_games, 4)}")
        print(f"Player 1 winrate: {round(player2_wins / num_unaugmented_games, 4)}\n")


def Train_NN(game_class, build_model_fn, configs, generation, folder_path, save_folder_path,
             run_dir):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    from Dataloader import Create_Dataset
    from Train import train

    gpu_devices = tf.config.list_physical_devices("GPU")
    for device in gpu_devices:
        # tf.config.experimental.set_virtual_device_configuration(device, [
        #     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5700)])
        tf.config.experimental.set_memory_growth(device, True)

    build_config, train_config, optimizer_config = configs
    # ^ these were previously read out of the module globals, which only resolves under
    # fork. Under spawn the __main__ guard never runs in the child, so they'd be undefined

    mixed_precision_policy = build_config.get("mixed_precision")
    if mixed_precision_policy is not None:
        policy = tf.keras.mixed_precision.Policy(mixed_precision_policy)
        tf.keras.mixed_precision.set_global_policy(policy)

    game = game_class()
    model = build_model_fn(game.get_input_state().shape, game.policy_shape, configs)
    model.load_weights(f"{folder_path}/model.weights.h5")

    try:
        learning_rate = optimizer_config["kwargs"]["learning_rate"](generation)
    except TypeError:
        learning_rate = optimizer_config["kwargs"]["learning_rate"]

    print(f"Started training for generation: {generation} using lr = {learning_rate}!")
    train_dataloader, test_dataloader = Create_Dataset(str(Path(folder_path).parent),
                                                       num_previous_generations=train_config["num_previous_generations"],
                                                       train_batch_size=train_config["train_batch_size"],
                                                       test_batch_size=train_config["test_batch_size"],
                                                       train_percent=train_config["train_percent"],
                                                       train_decay=train_config["train_decay"],
                                                       test_percent=train_config["test_percent"],
                                                       test_decay=train_config["test_decay"],
                                                       dedup_average=train_config.get("dedup_average", False), )
    model = train(train_dataloader, test_dataloader, model, learning_rate, configs,
                  run_dir=Path(run_dir).as_posix())
    Make_Generation_Folder(run_dir, generation + 1)
    model.save_weights(f"{save_folder_path}/model.weights.h5")


def Create_onnx(game_class, build_model_fn, configs, folder_path):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    build_config, _, _ = configs

    import tensorflow as tf
    from To_Onnx import convert_to_onnx

    game = game_class()
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
    print("Converting tensorflow model to onnx\n")
    input_signature = [tf.TensorSpec((None, *game.get_input_state().shape), tf.float32, name="inputs")]

    mixed_precision_policy = build_config.get("mixed_precision")
    if mixed_precision_policy is not None:
        policy = tf.keras.mixed_precision.Policy(mixed_precision_policy)
        tf.keras.mixed_precision.set_global_policy(policy)
    infer_model = build_model_fn(game.get_input_state().shape, game.policy_shape, configs)
    infer_model.load_weights(f"{folder_path}/model.weights.h5")
    convert_to_onnx(infer_model, input_signature, f"{folder_path}/model.onnx")
    print("Successfully converted to onnx\n")


def _initialize_model(game, build_model_fn, configs, folder_path):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    build_config, _, _ = configs

    import tensorflow as tf

    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            # tf.config.experimental.set_virtual_device_configuration(device, [
            #     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5700)])
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
    mixed_precision_policy = build_config.get("mixed_precision")
    if mixed_precision_policy is not None:
        policy = tf.keras.mixed_precision.Policy(mixed_precision_policy)
        tf.keras.mixed_precision.set_global_policy(policy)
    train_model = build_model_fn(game.get_input_state().shape, game.policy_shape, configs)
    train_model.summary()
    train_model.save_weights(f"{folder_path}/model.weights.h5")


def Initialize(game_class, build_model_fn, configs, run_dir):  # This must be ran with a mp.Process
    build_config, train_config, optimizer_config = configs
    # test the game class before anything is done
    print("\n*************Initiating*************\n")
    game = game_class()
    gen0 = gen_path(run_dir, 0)

    if mp.get_start_method() == "spawn":
        print("Running on Windows will cause massive slowdowns")
        print("Although Brian's code has to work around it, training on Linux is just much faster")
        print("Windows has to re spawn every process and thus requires to recreate everything")
        print(
            "Linux uses fork where the parent memory is just given to the new process and thus nothing need to be recreated")
        print("SO angry RN")
        print("L BOZO")
        print()
        print("Update: spawn was not actually the main villain here. The workers were")
        print("busy waiting on the shared memory flag instead of sleeping, so each one")
        print("burned a full core doing nothing and starved the very inference server it")
        print("was waiting on. Blocking on mp.Event instead took worker CPU from 140.6s")
        print("to 10.9s (12.9x) and wall time from 43.0s to 27.8s, and self play now")
        print("scales to ~12 workers instead of collapsing past 6.")
        print("The numba cache that kept needing deleting was not spawn's fault either,")
        print("it was the configs sharing a file with the njit kernels.")
        print("Windows is still slower. Just far less infuriating than it looked.")

    print("Initializing the model\n")
    p = mp.Process(target=_initialize_model, args=(game, build_model_fn, configs, gen0))
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError("Main process stopped")

    p = mp.Process(target=Create_onnx,
                   args=(game_class, build_model_fn, configs, gen0))
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError("Main process stopped")

    if train_config["use_tensorrt"]:
        p = mp.Process(target=cache_tensorrt, args=(game_class, configs, gen0))
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError("Main process stopped")
    Make_Dataset_File(gen0)


def Run(game_class, build_model_fn, configs, run_dir):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    build_config, train_config, optimizer_config = configs
    p = mp.Process(target=Validate_Configs, args=(configs,))
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError("Main process stopped")

    # the __pycache__ wipe that used to live here is gone. It was a workaround for the
    # numba cache being invalidated every run, which happened because the configs shared
    # a file with the njit kernels. config.py is separate now, so the cache is valid and
    # deleting it just bought a ~5s recompile in every worker

    if not Game_Tester(game_class).test():
        raise ValueError("Tests failed, training cannot continue!")

    os.makedirs(run_dir, exist_ok=True)
    tmp_best = Path(run_dir) / "tmp_best.weights.h5"
    if tmp_best.exists():
        os.remove(tmp_best)
    try:
        current_generation = max([int(Path(path).name)
                                  for path in glob(f"{Path(run_dir).as_posix()}/*")
                                  if Path(path).name.isdigit()])
    except ValueError:
        current_generation = 0

    if current_generation == 0:

        os.makedirs(gen_path(run_dir, 0), exist_ok=True)

        if (not os.path.exists(f"{gen_path(run_dir, 0)}/model.weights.h5") or not
        (os.path.exists(f"{gen_path(run_dir, 0)}/TRT_cache/model_ctx.onnx") if train_config["use_tensorrt"] else True) or not
        os.path.exists(f"{gen_path(run_dir, 0)}/model.onnx") or not
        os.path.exists(f"{gen_path(run_dir, 0)}/Self_Play_Data.h5")):
            print("Creating necessary files and models!")
            safe_rmtree(run_dir, run_dir)
            Make_Generation_Folder(run_dir, 0)

            Initialize(game_class, build_model_fn, configs, run_dir)
    # the main train loop

    # make model -> convert to onnx -> cache trt (optional) -> make dataset file
    #   loop:
    #       -> self play -> pad dataset -> train -> convert to onnx -> cache trt (optional) -> make dataset file

    # we have to get the current step in order to start loop training
    # finish the generation before loop training

    # we can continue on if we are still at the self play stage as none below will be True
    print("\n*************Starting*************\n")

    print(f"Generation: {current_generation} / {train_config['total_generations'] - 1}")

    current_path = gen_path(run_dir, current_generation)

    if "model.onnx" not in os.listdir(current_path):
        p = mp.Process(target=Create_onnx, args=(game_class, build_model_fn, configs,
                                                 current_path))
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError("Main process stopped")


    if "TRT_cache" not in os.listdir(current_path) and train_config["use_tensorrt"]:
        p = mp.Process(target=cache_tensorrt,
                       args=(game_class, configs, current_path))
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError("Main process stopped")

    p = mp.Process(target=compute_speed,
                   args=(game_class, configs, current_path))
    p.start()
    p.join()

    if p.exitcode != 0:
        raise RuntimeError("Main process stopped")

    if "Self_Play_Data.h5" not in os.listdir(current_path):
        Make_Dataset_File(current_path)
        current_generation += 1

    for generation in range(current_generation, train_config["total_generations"]):
        this_gen = gen_path(run_dir, generation)
        next_gen = gen_path(run_dir, generation + 1)

        safe_rmtree(f"{this_gen}/Cache", run_dir)
        run_self_play(game_class, configs, this_gen)
        safe_rmtree(f"{this_gen}/Cache", run_dir)
        Print_Stats(this_gen)

        p = mp.Process(target=Train_NN, args=(game_class, build_model_fn, configs, generation,
                                              this_gen, next_gen, run_dir))
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError("Main process stopped")

        p = mp.Process(target=Create_onnx,
                       args=(
                           game_class, build_model_fn, configs, next_gen))
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError("Main process stopped")

        if train_config["use_tensorrt"]:
            p = mp.Process(target=cache_tensorrt,
                           args=(game_class, configs, next_gen))
            p.start()
            p.join()
            if p.exitcode != 0:
                raise RuntimeError("Main process stopped")

            p = mp.Process(target=compute_speed,
                           args=(game_class, configs, next_gen))
            p.start()
            p.join()
            if p.exitcode != 0:
                raise RuntimeError("Main process stopped")
        if generation < train_config["total_generations"] - 1:
            Make_Dataset_File(next_gen)
            print(f"Generation: {generation + 1} / {train_config['total_generations'] - 1}")
    print("-----------Training Done!-----------")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", default=DEFAULT_GAME,
                        help=f"game package to train (default: {DEFAULT_GAME})")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()

    game_class, build_model, configs, run_dir = load_game(args.game)
    print(f"Training {args.game}, output goes to {run_dir}")
    Run(game_class, build_model, configs, run_dir)
