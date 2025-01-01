import os
import numpy as np
import h5py as h5
import tensorflow as tf
import multiprocessing as mp

from Gomoku import Gomoku, build_config, train_config
from to_onnx import convert_to_onnx
from Build_Model import build_model, build_model_infer
from Build_Tensorrt import cache_tensorrt, get_speed
from Self_Play import run_self_play

def validate_train_config(train_config):
    if train_config["total_generations"] <= 0:
        raise ValueError("Total generations can't less than or equal to zero")

    if train_config["games_per_generation"] <= 0:
        raise ValueError("Games per generation can't be less than or equal to zero")

    if not train_config["use_gpu"] and train_config["use_tensorrt"]:
        raise ValueError("You must use the GPU for tensorrt")
def initialize_training(): # This must be ran with a mp.Process
    os.makedirs("Grok_Zero_Train/0/", exist_ok=True)

    if os.path.exists("Grok_Zero_Train/0/model.keras") and not os.path.exists("Grok_Zero_Train/0/TRT_cache/model_ctx.onnx")and not os.path.exists("Grok_Zero_Train/0/model.onnx"):
        raise ValueError("Please delete the entire Grok_Zero_Train folder and start the training again!")

    validate_train_config(train_config)

    game= Gomoku()
    def initialize_model(game, build_config):
        train_model = build_model(game.get_input_state().shape, game.policy_shape, build_config)
        train_model.save("Grok_Zero_Train/0/model.keras")

    p = mp.Process(target=initialize_model, args=(game, build_config))
    p.start()
    p.join()

    def initialize_onnx(build_config):
        print("Converting tensorflow model to onnx")
        embed_size, num_heads, num_layers = build_config["embed_size"], build_config["num_heads"], build_config["num_layers"]
        input_signature = [tf.TensorSpec((1, *game.get_input_state().shape), tf.float32, name="inputs"),
                           tf.TensorSpec((num_layers, 2, 1, embed_size), tf.float32, name="input_state"),
                           tf.TensorSpec((num_layers, 1, num_heads, embed_size // num_heads, embed_size // num_heads),
                                         tf.float32, name="input_state_matrix"),
                           ]
        infer_model = build_model_infer(game.get_input_state().shape, game.policy_shape, build_config)
        infer_model.load_weights("Grok_Zero_Train/0/model.keras")
        convert_to_onnx(infer_model, input_signature, "Grok_Zero_Train/0/model.onnx")
        print("Successfully converted to onnx")

    p = mp.Process(target=initialize_onnx, args=(build_config,))
    p.start()
    p.join()

    if train_config["use_gpu"] and train_config["use_tensorrt"]:
        p = mp.Process(target=cache_tensorrt, args=(game, build_config, "Grok_Zero_Train/", 0))
        p.start()
        p.join()

        p = mp.Process(target=get_speed, args=(game, build_config, "Grok_Zero_Train/", 0))
        p.start()
        p.join()

    with h5.File("Grok_Zero_Train/0/Self_Play_Data.h5", "w", libver="latest") as file:
        file.create_dataset(f"max_moves", maxshape=(1,), dtype=np.int32, data=np.zeros(1,))

if __name__ == "__main__":
    initialize_training()