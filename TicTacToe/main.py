import os
import shutil

from glob import glob
import h5py as h5
import numpy as np
import tensorflow as tf
import multiprocessing as mp

from Tictactoe import TicTacToe, build_config, train_config
from Game_Tester import Game_Tester

from To_onnx import convert_to_onnx
from Build_Model import build_model, build_model_infer
from Build_Tensorrt import cache_tensorrt, get_speed


def Validate_Train_Config(train_config):
    if train_config["total_generations"] <= 0:
        raise ValueError("Total generations can't less than or equal to zero")

    if train_config["games_per_generation"] < 2:
        raise ValueError("Games per generation can't be less than 2")

    if not train_config["use_gpu"] and train_config["use_tensorrt"]:
        raise ValueError("You must use the GPU for tensorrt")

def Make_Generation_Folder(generation):
    os.makedirs(f"Grok_Zero_Train/{generation}/", exist_ok=False)
def Make_Dataset_File(folder_path):
    with h5.File(f"{folder_path}/Self_Play_Data.h5", "w", libver="latest") as file:
        file.create_dataset(f"max_actions", maxshape=(1,), dtype=np.int32, data=np.zeros(1,))
        file.create_dataset(f"num_unaugmented_games", maxshape=(1,), dtype=np.int32, data=np.zeros(1,))

def Initialize_Training(game_class): # This must be ran with a mp.Process
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


    def initialize_onnx(game_class, build_config):
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
        infer_model.load_weights("Grok_Zero_Train/0/model.weights.h5")
        convert_to_onnx(infer_model, input_signature, "Grok_Zero_Train/0/model.onnx")
        print("Successfully converted to onnx\n")

    p = mp.Process(target=initialize_onnx, args=(game_class, build_config,))
    p.start()
    p.join()


    if train_config["use_gpu"] and train_config["use_tensorrt"]:
        p = mp.Process(target=cache_tensorrt, args=(game, build_config, train_config, "Grok_Zero_Train/", 0))
        p.start()
        p.join()

        p = mp.Process(target=get_speed, args=(game, build_config, train_config, "Grok_Zero_Train/", 0))
        p.start()
        p.join()

    Make_Dataset_File("Grok_Zero_Train/0/")

def Run(game_class, test=False):
    Validate_Train_Config(train_config)
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

            Initialize_Training(game_class)


if __name__ == "__main__":
    Run(TicTacToe, test=True)