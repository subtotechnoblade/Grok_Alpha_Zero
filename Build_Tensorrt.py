import time
import numpy as np
import onnxruntime as rt
# This assumes that use_tensorrt is True in train config
def convert_shape(shape):
    assert len(shape) > 0
    str_shape = ""
    for dim in shape:
        str_shape += f"{dim}x"
    return str_shape[:-1]

def cache_tensorrt(game_class,
                   build_config,
                   train_config,
                   folder_path,
                   warmup_iterations=10):
    # uses the generation number to build the generation cache
    embed_size = build_config["embed_size"]
    num_heads = build_config["num_heads"]
    num_layers = build_config["num_layers"]

    game = game_class()
    board_shape = game.board.shape
    str_board_shape = convert_shape(board_shape)
    max_shape = train_config["num_workers"]


    providers = [
        ('TensorrtExecutionProvider', {
            "trt_engine_cache_enable": True,
            "trt_dump_ep_context_model": True,
            "trt_builder_optimization_level": 5,
            "trt_auxiliary_streams": 0,

            "trt_ep_context_file_path": f"{folder_path}/TRT_cache/",
            "trt_profile_min_shapes": f"inputs:1x{str_board_shape},input_state:{num_layers}x2x1x{embed_size},input_state_matrix:{num_layers}x1x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",
            "trt_profile_max_shapes": f"inputs:{max_shape}x{str_board_shape},input_state:{num_layers}x2x{max_shape}x{embed_size},input_state_matrix:{num_layers}x{max_shape}x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",
            "trt_profile_opt_shapes": f"inputs:{max_shape}x{str_board_shape},input_state:{num_layers}x2x{max_shape}x{embed_size},input_state_matrix:{num_layers}x{max_shape}x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",
        }),
        'CUDAExecutionProvider',
        'CPUExecutionProvider']

    print("Building and caching tensorrt engine!")

    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = rt.InferenceSession(f"{folder_path}/model.onnx",
                               sess_options=sess_options,
                               providers=providers)


    dummy_inputs = np.random.uniform(-1, 2, (1, *game.get_input_state().shape)).astype(np.float32)

    input_state = np.zeros((num_layers, 2, 1, embed_size), dtype=np.float32)
    input_state_matrix = np.zeros((num_layers, 1, num_heads, embed_size // num_heads, embed_size // num_heads), dtype=np.float32)
    for _ in range(warmup_iterations):
        policy, value, state, state_matrix = sess.run(["policy", "value", "output_state", "output_state_matrix"],
                                                              input_feed={"inputs": dummy_inputs,
                                                                "input_state": input_state,
                                                                "input_state_matrix": input_state_matrix})
    print("Successfully built trt engine!")
