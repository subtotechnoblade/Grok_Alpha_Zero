from pathlib import Path

import numpy as np
# This assumes that use_tensorrt is True in train config
def convert_shape(shape):
    assert len(shape) > 0
    str_shape = ""
    for dim in shape:
        str_shape += f"{dim}x"
    return str_shape[:-1]

def cache_tensorrt(game_class,
                   configs,
                   folder_path,
                   warmup_iterations=100):
    import onnxruntime as rt
    from Client_Server import preload_gpu_dlls
    preload_gpu_dlls()  # windows: without this the TensorRT provider DLL fails to
    # load, the engine is never written, and self play then dies looking for it
    # uses the generation number to build the generation cache
    build_config, train_config, optimizer_config = configs

    game = game_class()
    str_board_shape = convert_shape(game.get_input_state().shape)
    leaves_per_batch = train_config.get("leaves_per_batch", 1) if not train_config.get("use_gumbel") else 1
    max_shape = train_config["num_workers"] * leaves_per_batch


    providers = [
        ('TensorrtExecutionProvider', {
            "trt_engine_cache_enable": True,
            "trt_dump_ep_context_model": True,
            "trt_builder_optimization_level": 5,
            "trt_auxiliary_streams": 0,
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": f"{Path(folder_path).parent}/",

            "trt_ep_context_file_path": f"{folder_path}/TRT_cache/",
            "trt_profile_min_shapes": f"inputs:1x{str_board_shape}",
            "trt_profile_max_shapes": f"inputs:{max_shape}x{str_board_shape}",
            "trt_profile_opt_shapes": f"inputs:{max_shape}x{str_board_shape}",
        }),
        'CUDAExecutionProvider',
        'CPUExecutionProvider']
    if train_config.get("trt_fp16",
                        build_config.get("mixed_precision") == "mixed_float16"):
        providers[0][1]["trt_fp16_enable"] = True

    print("Building and caching tensorrt engine!")

    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = rt.InferenceSession(f"{folder_path}/model.onnx",
                               sess_options=sess_options,
                               providers=providers)


    from Client_Server import session_input_dtypes
    dummy_inputs = np.random.randint(0, 2, (1, *game.get_input_state().shape)).astype(
        next(iter(session_input_dtypes(sess).values())))

    for _ in range(warmup_iterations):
        policy, value = sess.run(["policy", "value"],
                                                              input_feed={"inputs": dummy_inputs})
    print("Successfully built trt engine!")
