import time
import numpy as np
import onnxruntime as rt

def compute_speed(game_class,
              build_config,
              train_config,
              folder_path,
              iterations=500):
    game = game_class()

    embed_size = build_config["embed_size"]
    num_heads = build_config["num_heads"]
    num_layers = build_config["num_layers"]

    max_shape = opt_shape = train_config["num_workers"]
    providers = [
        ('TensorrtExecutionProvider', {
        "trt_engine_cache_enable":True,
        "trt_dump_ep_context_model": True,
        "trt_builder_optimization_level": 5,
        "trt_auxiliary_streams": 0,
        "trt_ep_context_file_path": f"{folder_path}/TRT_cache/",

        "trt_profile_min_shapes": f"inputs:1x15x15,input_state:{num_layers}x2x1x{embed_size},input_state_matrix:{num_layers}x1x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",
        "trt_profile_max_shapes": f"inputs:{max_shape}x15x15,input_state:{num_layers}x2x{max_shape}x{embed_size},input_state_matrix:{num_layers}x{max_shape}x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",
        "trt_profile_opt_shapes": f"inputs:{opt_shape}x15x15,input_state:{num_layers}x2x{opt_shape}x{embed_size},input_state_matrix:{num_layers}x{opt_shape}x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",

        }),
        'CUDAExecutionProvider',
        'CPUExecutionProvider']


    sess = rt.InferenceSession(f"{folder_path}/TRT_cache/model_ctx.onnx",
                               providers=providers)

    dummy_inputs = np.random.uniform(-1, 2, (train_config["num_workers"], *game.get_input_state().shape)).astype(np.float32)

    input_state = np.zeros((num_layers, 2, train_config["num_workers"], embed_size), dtype=np.float32)
    input_state_matrix = np.zeros((num_layers, train_config["num_workers"], num_heads, embed_size // num_heads, embed_size // num_heads), dtype=np.float32)

    for _ in range(10):
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
    print(iterations)
    print(f"Took {time_taken / iterations} seconds per iteration at {iterations / time_taken:0.2f} it/s!")