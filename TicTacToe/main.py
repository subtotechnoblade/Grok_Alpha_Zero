import os
from pathlib import Path
import shutil

import h5py as h5
import numpy as np
from glob import glob
import tensorflow as tf
import onnxruntime as rt
import multiprocessing as mp

from Build_Model import build_model, build_model_infer
from Build_Tensorrt import cache_tensorrt, get_speed

from Game_Tester import Game_Tester

from Self_Play import run_self_play
from Pad_Dataset import Pad_Dataset
from Train import train
from To_onnx import convert_to_onnx


def Validate_Train_Config(train_config):
    if train_config["total_generations"] <= 0:
        raise ValueError("Total generations can't less than or equal to zero")

    if train_config["games_per_generation"] < 2:
        raise ValueError("Games per generation can't be less than 2")

    if not train_config["use_gpu"] and train_config["use_tensorrt"]:
        raise ValueError("You must use the GPU for tensorrt")

    available_providers = rt.get_available_providers()
    if train_config["use_tensorrt"] and "TensorrtExecutionProvider" not in available_providers:
        raise RuntimeError("Please install tensorrt as onnxruntime doesn't detect TensorrtExecutionProvider")

    if train_config["use_gpu"] and not train_config["use_tensorrt"] and "CUDAExecutionProvider" not in available_providers:
        raise RuntimeError("Please install CUDA as onnxruntime doesn't detect CUDAExecutionProvider")

    if train_config["optimizer"].lower() not in ["adam", "adamw", "nadam"]:
        raise ValueError(f"Optimizer must be either Adam, AdamW, or Nadam got {train_config['optimizer']}.")


def Make_Generation_Folder(generation):
    os.makedirs(f"Grok_Zero_Train/{generation}/", exist_ok=False)
def Make_Dataset_File(folder_path):
    with h5.File(f"{folder_path}/Self_Play_Data.h5", "w", libver="latest") as file:
        file.create_dataset(f"game_stats", maxshape=(6,), dtype=np.uint32, data=np.zeros(6,))
        # max_actions, total_actions, num_unaugmented_games, player -1 wins, draws, player 1 wins
def Train_NN(game_class, build_config, train_config, generation, folder_path, save_folder_path):
    game = game_class()
    model = build_model(game.get_input_state().shape, game.policy_shape, build_config)
    model.load_weights(f"{folder_path}/model.weights.h5")

    lr_decay = train_config["lr_decay"] ** (generation // train_config["decay_lr_after"])
    learning_rate = train_config["learning_rate"] * lr_decay

    print(f"Started training for generation: {generation} using lr = {learning_rate}!")
    train(model, learning_rate, train_config, folder_path, save_folder_path)
def Create_onnx(game_class, build_config, folder_path):
    game = game_class()
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
    print("Converting tensorflow model to onnx\n")
    embed_size, num_heads, num_layers = build_config["embed_size"], build_config["num_heads"], build_config["num_layers"]
    input_signature = [tf.TensorSpec((None, *game.get_input_state().shape), tf.float32, name="inputs"),
                       tf.TensorSpec((num_layers, 2, None, embed_size), tf.float32, name="input_state"),
                       tf.TensorSpec((num_layers, None, num_heads, embed_size // num_heads, embed_size // num_heads),
                                     tf.float32, name="input_state_matrix"),
                       ]
    infer_model = build_model_infer(game.get_input_state().shape, game.policy_shape, build_config)
    infer_model.load_weights(f"{folder_path}/model.weights.h5")
    convert_to_onnx(infer_model, input_signature, f"{folder_path}/model.onnx")
    print("Successfully converted to onnx\n")
def Initialize(game_class, build_config, train_config): # This must be ran with a mp.Process
    # test the game class before anything is done
    print("\n*************Initiating*************\n")
    game = game_class()
    def initialize_model(game, build_config):
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass
        train_model = build_model(game.get_input_state().shape, game.policy_shape, build_config)
        train_model.save_weights("Grok_Zero_Train/0/model.weights.h5")
    print("Initializing the model\n")
    p = mp.Process(target=initialize_model, args=(game, build_config))
    p.start()
    p.join()


    p = mp.Process(target=Create_onnx, args=(game_class, build_config, "Grok_Zero_Train/0"))
    p.start()
    p.join()


    if train_config["use_tensorrt"]:
        p = mp.Process(target=cache_tensorrt, args=(game_class, build_config, train_config, "Grok_Zero_Train/0"))
        p.start()
        p.join()

        p = mp.Process(target=get_speed, args=(game_class, build_config, train_config, "Grok_Zero_Train/0"))
        p.start()
        p.join()

    Make_Dataset_File("Grok_Zero_Train/0/")

def Run(game_class, build_config, train_config, test=False):
    Validate_Train_Config(train_config)

    parent_dir = Path(__file__).resolve().parent # delete pycache in the parent directory
    if "__pycache__" in os.listdir(parent_dir):
        shutil.rmtree(f"{parent_dir}/__pycache__")

    if test:
        Game_Tester(game_class).test()
    try:
        current_generation = max([int(path.split("/")[-1]) for path in glob("Grok_Zero_Train/*")])
    except:
        current_generation = 0

    if current_generation == 0:

        os.makedirs("Grok_Zero_Train/0/", exist_ok=True)

        if (not os.path.exists("Grok_Zero_Train/0/model.weights.h5") or not
        (train_config["use_tensorrt"] and os.path.exists("Grok_Zero_Train/0/TRT_cache/model_ctx.onnx")) or not
        os.path.exists("Grok_Zero_Train/0/model.onnx") or not
        os.path.exists("Grok_Zero_Train/Self_Play_Data.h5")):
            print("The starting folder is initialized incorrectly!")
            print("Deleting and re-initiating!\n")
            shutil.rmtree("Grok_Zero_Train/")
            Make_Generation_Folder(0)

            Initialize(game_class, build_config, train_config)

    # the main train loop

    # make model -> convert to onnx -> cache trt (optional) -> make dataset file
    #   loop:
    #       -> self play -> pad dataset -> train -> convert to onnx -> cache trt (optional) -> make dataset file

    # we have to get the current step in order to start loop training
    # finish the generation before loop training

    # we can continue on if we are still at the self play stage as none below will be True
    if len(os.listdir(f"Grok_Zero_Train/{current_generation}")) == 0: # on the training step
        p = mp.Process(target=Train_NN, args=(game_class, build_config, train_config, current_generation,
                                              f"Grok_Zero_Train/{current_generation - 1}",
                                              f"Grok_Zero_Train/{current_generation}"))
        p.start()
        p.join()

    if "model.onnx" not in os.listdir(f"Grok_Zero_Train/{current_generation}"):
        p = mp.Process(target=Create_onnx, args=(game_class, build_config,
                                                 f"Grok_Zero_Train/{current_generation}"))
        p.start()
        p.join()

    if train_config["use_tensorrt"]:
        if "TRT_cache" not in os.listdir(f"Grok_Zero_Train/{current_generation}"):
            p = mp.Process(target=cache_tensorrt, args=(game_class, build_config, train_config, f"Grok_Zero_Train/{current_generation}"))
            p.start()
            p.join()

            p = mp.Process(target=get_speed, args=(game_class, build_config, train_config, f"Grok_Zero_Train/{current_generation}"))
            p.start()
            p.join()

    if "Self_Play_Data.h5" not in os.listdir(f"Grok_Zero_Train/{current_generation}"):
        Make_Dataset_File(f"Grok_Zero_Train/{current_generation}/")



    # for generation in range(current_generation, train_config["total_generations"]):
    #     run_self_play(game_class, build_config, train_config, "Grok_Zero_Train/")


if __name__ == "__main__":
    from Tictactoe import TicTacToe, build_config, train_config
    Run(TicTacToe, build_config, train_config, test=False)