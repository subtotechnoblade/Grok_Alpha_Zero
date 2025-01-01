import time
import numpy as np
import onnxruntime as rt
# This assumes that use_tensorrt is True in train config

def cache_tensorrt(game,
                   build_config,
                   folder_path,
                   generation,
                   warmup_iterations=10):
    # uses the generation number to build the generation cache
    providers = [
        ('TensorrtExecutionProvider', {
        "trt_engine_cache_enable":True,
        "trt_dump_ep_context_model": True,
        "trt_builder_optimization_level": 5,
        "trt_auxiliary_streams": 0,
        "trt_ep_context_file_path": f"{folder_path}/{generation}/TRT_cache/"
        }),
        'CUDAExecutionProvider',
        'CPUExecutionProvider']

    print("Building and caching tensorrt engine!")

    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = rt.InferenceSession(f"{folder_path}/{generation}/model.onnx",
                               sess_options=sess_options,
                               providers=providers)


    dummy_inputs = np.random.uniform(-1, 2, (1, *game.get_input_state().shape)).astype(np.float32)

    embed_size = build_config["embed_size"]
    num_heads = build_config["num_heads"]
    num_layers = build_config["num_layers"]

    input_state = np.zeros((num_layers, 2, 1, embed_size), dtype=np.float32)
    input_state_matrix = np.zeros((num_layers, 1, num_heads, embed_size // num_heads, embed_size // num_heads), dtype=np.float32)
    for _ in range(warmup_iterations):
        policy, value, state, state_matrix = sess.run(["policy", "value", "output_state", "output_state_matrix"],
                                                              input_feed={"inputs": dummy_inputs,
                                                                "input_state": input_state,
                                                                "input_state_matrix": input_state_matrix})
    print("Successfully built trt engine!")

def get_speed(game, build_config, folder_path, generation, iterations=100):
    providers = [
        ('TensorrtExecutionProvider', {
        "trt_engine_cache_enable":True,
        "trt_dump_ep_context_model": True,
        "trt_builder_optimization_level": 5,
        "trt_auxiliary_streams": 0,
        "trt_ep_context_file_path": f"{folder_path}/{generation}/TRT_cache/"
        }),
        'CUDAExecutionProvider',
        'CPUExecutionProvider']


    sess = rt.InferenceSession(f"{folder_path}/{generation}/TRT_cache/model_ctx.onnx",
                               providers=providers)

    dummy_inputs = np.random.uniform(-1, 2, (1, *game.get_input_state().shape)).astype(np.float32)

    embed_size = build_config["embed_size"]
    num_heads = build_config["num_heads"]
    num_layers = build_config["num_layers"]

    input_state = np.zeros((num_layers, 2, 1, embed_size), dtype=np.float32)
    input_state_matrix = np.zeros((num_layers, 1, num_heads, embed_size // num_heads, embed_size // num_heads), dtype=np.float32)

    for _ in range(5):
        policy, value, state, state_matrix = sess.run(["policy", "value", "output_state", "output_state_matrix"],
                                                              input_feed={"inputs": dummy_inputs,
                                                                "input_state": input_state,
                                                                "input_state_matrix": input_state_matrix})

    s = time.time()
    for _ in range(iterations):
        policy, value, state, state_matrix = sess.run(["policy", "value", "output_state", "output_state_matrix"],
                                                              input_feed={"inputs": dummy_inputs,
                                                                "input_state": input_state,
                                                                "input_state_matrix": input_state_matrix})
    time_taken = time.time() - s
    print(f"Took {time_taken / 100} per iteration at {100 / time_taken} it/s!")