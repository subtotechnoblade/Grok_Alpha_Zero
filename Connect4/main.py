import os
import shutil
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


def Validate_Train_Config(train_config):
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
        "use_tensorrt"] and "CUDAExecutionProvider" not in available_providers:
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

    mixed_precision_policy = train_config.get("mixed_precision", None)
    if mixed_precision_policy is not None:
        if mixed_precision_policy == "mixed_bfloat16":
            raise ValueError("mixed_bfloat16 isn't supported, its not because of me, blame tf2onnx for not supporting it, I have no workaround")
        elif mixed_precision_policy != "mixed_float16":
            raise ValueError(
                f"mixed_precision param is invalid got: {mixed_precision_policy}, should be None, mixed_float16, mixed_bfloat16")
    if mixed_precision_policy == "mixed_bfloat16" and not train_config["use_gpu"]:
        warnings.warn("Using float16 as the compute type for CPU is extremely slow!")
    if train_config["optimizer"].lower() not in ["adam", "adamw", "nadam"]:
        raise ValueError(f"Optimizer must be either Adam, AdamW, or Nadam got {train_config['optimizer']}.")


def Make_Generation_Folder(generation):
    os.makedirs(f"Grok_Zero_Train/{generation}/", exist_ok=False)


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


def Train_NN(game_class, build_model_fn, build_config, train_config, generation, folder_path, save_folder_path):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    from Dataloader import Create_Dataset
    from Train import train

    gpu_devices = tf.config.list_physical_devices("GPU")
    for device in gpu_devices:
        # tf.config.experimental.set_virtual_device_configuration(device, [
        #     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5700)])
        tf.config.experimental.set_memory_growth(device, True)

    mixed_precision_policy = train_config.get("mixed_precision")
    if mixed_precision_policy is not None:
        policy = tf.keras.mixed_precision.Policy(mixed_precision_policy)
        tf.keras.mixed_precision.set_global_policy(policy)

    game = game_class()
    model = build_model_fn(game.get_input_state().shape, game.policy_shape, build_config, train_config)
    model.load_weights(f"{folder_path}/model.weights.h5")

    lr_decay = train_config["lr_decay"] ** (generation // train_config["decay_lr_after"])
    learning_rate = train_config["learning_rate"] * lr_decay

    print(f"Started training for generation: {generation} using lr = {learning_rate}!")
    train_dataloader, test_dataloader = Create_Dataset(str(Path(folder_path).parent),
                                                       num_previous_generations=train_config["num_previous_generations"],
                                                       train_batch_size=train_config["train_batch_size"],
                                                       test_batch_size=train_config["test_batch_size"],
                                                       train_percent=train_config["train_percent"],
                                                       train_decay=train_config["train_decay"],
                                                       test_percent=train_config["test_percent"],
                                                       test_decay=train_config["test_decay"], )
    model = train(train_dataloader, test_dataloader, model, learning_rate, build_config, train_config)
    Make_Generation_Folder(generation + 1)
    model.save_weights(f"{save_folder_path}/model.weights.h5")


def Create_onnx(game_class, build_model_fn, build_config, train_config, folder_path):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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

    mixed_precision_policy = train_config.get("mixed_precision")
    if mixed_precision_policy is not None:
        policy = tf.keras.mixed_precision.Policy(mixed_precision_policy)
        tf.keras.mixed_precision.set_global_policy(policy)
    infer_model = build_model_fn(game.get_input_state().shape, game.policy_shape, build_config, train_config)
    infer_model.load_weights(f"{folder_path}/model.weights.h5")
    convert_to_onnx(infer_model, input_signature, f"{folder_path}/model.onnx")
    print("Successfully converted to onnx\n")


def _initialize_model(game, build_model_fn, build_config, train_config):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
    mixed_precision_policy = train_config.get("mixed_precision")
    if mixed_precision_policy is not None:
        policy = tf.keras.mixed_precision.Policy(mixed_precision_policy)
        tf.keras.mixed_precision.set_global_policy(policy)
    train_model = build_model_fn(game.get_input_state().shape, game.policy_shape, build_config, train_config)
    train_model.summary()
    train_model.save_weights("Grok_Zero_Train/0/model.weights.h5")


def Initialize(game_class, build_model_fn, build_config, train_config):  # This must be ran with a mp.Process
    # test the game class before anything is done
    print("\n*************Initiating*************\n")
    game = game_class()

    if mp.get_start_method() == "spawn":
        print("Running on Windows will cause massive slowdowns")
        print("Although Brian's code has to work around it, training on Linux is just much faster")
        print("Windows has to re spawn every process and thus requires to recreate everything")
        print(
            "Linux uses fork where the parent memory is just given to the new process and thus nothing need to be recreated")
        print("SO angry RN")
        print("L BOZO")

    print("Initializing the model\n")
    p = mp.Process(target=_initialize_model, args=(game, build_model_fn, build_config, train_config))
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError("Main process stopped")

    p = mp.Process(target=Create_onnx,
                   args=(game_class, build_model_fn, build_config, train_config, "Grok_Zero_Train/0"))
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError("Main process stopped")

    if train_config["use_tensorrt"]:
        p = mp.Process(target=cache_tensorrt, args=(game_class, build_config, train_config, "Grok_Zero_Train/0"))
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError("Main process stopped")

    p = mp.Process(target=compute_speed, args=(game_class, build_config, train_config, "Grok_Zero_Train/0"))
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError("Main process stopped")

    Make_Dataset_File("Grok_Zero_Train/0")


def Run(game_class, build_model_fn, build_config, train_config):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    p = mp.Process(target=Validate_Train_Config, args=(train_config,))
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError("Main process stopped")

    parent_dir = Path(__file__).resolve().parent  # delete pycache in the parent directory
    if "__pycache__" in os.listdir(parent_dir):
        shutil.rmtree(f"{parent_dir}/__pycache__")

    if not Game_Tester(game_class).test():
        raise ValueError("Tests failed, training cannot continue!")

    if os.path.exists("Grok_Zero_Train/tmp_best.weights.h5"):
        os.remove("Grok_Zero_Train/tmp_best.weights.h5")
    try:
        current_generation = max([int(Path(path).name) for path in glob("Grok_Zero_Train/*")])
    except:
        current_generation = 0

    if current_generation == 0:

        os.makedirs("Grok_Zero_Train/0/", exist_ok=True)

        if (not os.path.exists("Grok_Zero_Train/0/model.weights.h5") or not
        (os.path.exists("Grok_Zero_Train/0/TRT_cache/model_ctx.onnx") if train_config["use_tensorrt"] else True) or not
        os.path.exists("Grok_Zero_Train/0/model.onnx") or not
        os.path.exists("Grok_Zero_Train/0/Self_Play_Data.h5")):
            print("Creating necessary files and models!")
            shutil.rmtree("Grok_Zero_Train/")
            Make_Generation_Folder(0)

            Initialize(game_class, build_model_fn, build_config, train_config)
    # the main train loop

    # make model -> convert to onnx -> cache trt (optional) -> make dataset file
    #   loop:
    #       -> self play -> pad dataset -> train -> convert to onnx -> cache trt (optional) -> make dataset file

    # we have to get the current step in order to start loop training
    # finish the generation before loop training

    # we can continue on if we are still at the self play stage as none below will be True
    print("\n*************Starting*************\n")

    print(f"Generation: {current_generation} / {train_config['total_generations'] - 1}")

    calculate_speed = False
    if "model.onnx" not in os.listdir(f"Grok_Zero_Train/{current_generation}"):
        p = mp.Process(target=Create_onnx, args=(game_class, build_model_fn, build_config, train_config,
                                                 f"Grok_Zero_Train/{current_generation}"))
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError("Main process stopped")
        calculate_speed = True

    if "TRT_cache" not in os.listdir(f"Grok_Zero_Train/{current_generation}") and train_config["use_tensorrt"]:
        p = mp.Process(target=cache_tensorrt,
                       args=(game_class, build_config, train_config, f"Grok_Zero_Train/{current_generation}"))
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError("Main process stopped")
        calculate_speed = True

    if calculate_speed:
        p = mp.Process(target=compute_speed,
                       args=(game_class, build_config, train_config, f"Grok_Zero_Train/{current_generation}"))
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError("Main process stopped")

    if "Self_Play_Data.h5" not in os.listdir(f"Grok_Zero_Train/{current_generation}"):
        Make_Dataset_File(f"Grok_Zero_Train/{current_generation}/")

        current_generation += 1

    for generation in range(current_generation, train_config["total_generations"]):
        if os.path.exists(f"Grok_Zero_Train/{generation}/Cache"):
            shutil.rmtree(f"Grok_Zero_Train/{generation}/Cache")
        run_self_play(game_class, build_config, train_config, f"Grok_Zero_Train/{generation}")
        if os.path.exists(f"Grok_Zero_Train/{generation}/Cache"):
            shutil.rmtree(f"Grok_Zero_Train/{generation}/Cache")
        Print_Stats(f"Grok_Zero_Train/{generation}")

        p = mp.Process(target=Train_NN, args=(game_class, build_model_fn, build_config, train_config, generation,
                                              f"Grok_Zero_Train/{generation}", f"Grok_Zero_Train/{generation + 1}"))
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError("Main process stopped")

        p = mp.Process(target=Create_onnx,
                       args=(
                           game_class, build_model_fn, build_config, train_config, f"Grok_Zero_Train/{generation + 1}"))
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError("Main process stopped")

        if train_config["use_tensorrt"]:
            p = mp.Process(target=cache_tensorrt,
                           args=(game_class, build_config, train_config, f"Grok_Zero_Train/{generation + 1}"))
            p.start()
            p.join()
            if p.exitcode != 0:
                raise RuntimeError("Main process stopped")

            p = mp.Process(target=compute_speed,
                           args=(game_class, build_config, train_config, f"Grok_Zero_Train/{generation + 1}"))
            p.start()
            p.join()
            if p.exitcode != 0:
                raise RuntimeError("Main process stopped")
        if generation < train_config["total_generations"] - 1:
            Make_Dataset_File(f"Grok_Zero_Train/{generation + 1}")
            print(f"Generation: {generation + 1} / {train_config['total_generations'] - 1}")
    print("-----------Training Done!-----------")


if __name__ == "__main__":
    from Connect4 import Connect4, build_config, train_config
    from Build_Model import build_model

    Run(Connect4, build_model, build_config, train_config)
