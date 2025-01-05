import time
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

from Client_Server import Parallelized_Session, start_server, create_shared_memory, convert_to_single_info


if __name__ == "__main__":
    from Gomoku.Gomoku import build_config, train_config
    embed_size, num_heads, num_layers = build_config["embed_size"],  build_config["num_heads"], build_config["num_layers"]
    max_shape, opt_shape = 12, 12
    providers = [
        ('TensorrtExecutionProvider', {
            "trt_engine_cache_enable": True,
            "trt_dump_ep_context_model": True,
            "trt_builder_optimization_level": 5,
            "trt_auxiliary_streams": 0,
            "trt_ep_context_file_path": "Gomoku/Cache/",

            "trt_profile_min_shapes": f"inputs:1x15x15,input_state:{num_layers}x2x1x{embed_size},input_state_matrix:{num_layers}x1x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",
            "trt_profile_max_shapes": f"inputs:{max_shape}x15x15,input_state:{num_layers}x2x{max_shape}x{embed_size},input_state_matrix:{num_layers}x{max_shape}x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",
            "trt_profile_opt_shapes": f"inputs:{opt_shape}x15x15,input_state:{num_layers}x2x{opt_shape}x{embed_size},input_state_matrix:{num_layers}x{opt_shape}x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",
        }),
        'CUDAExecutionProvider',
        'CPUExecutionProvider']
    # sess_options = rt.SessionOptions()
    # sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

    batched_inputs_feed_info = {"inputs": [[-1, 15, 15], np.float32],
                                "input_state": [[num_layers, 2, -1, embed_size], np.float32],
                                "input_state_matrix": [[num_layers, -1, num_heads, embed_size // num_heads, embed_size // num_heads], np.float32]
                                }
    infer_input_feed_info = convert_to_single_info(batched_inputs_feed_info)
    batched_outputs_feed_info = {"policy": [-1, 225],
                                 "value": [-1, 1],
                                 "output_state": [num_layers, 2, -1, embed_size],
                                 "output_state_matrix": [num_layers, -1, num_heads, embed_size // num_heads, embed_size // num_heads]
                                 }
    infer_output_feed_info = convert_to_single_info(batched_outputs_feed_info)

    num_workers = 9
    shms = create_shared_memory(batched_inputs_feed_info, batched_outputs_feed_info, num_workers=num_workers)

    def slam(slammer_id,
             infer_input_feed_info,
             infer_output_feed_info,
             shm):
        dummy_inputs = np.random.randint(-1, 2, size=(1, 15, 15))
        dummy_state = np.random.uniform(size=[num_layers, 2, 1, embed_size]).astype(dtype=np.float32)
        dummy_state_matrix = np.random.uniform(
            size=[num_layers, 1, num_heads, embed_size // num_heads, embed_size // num_heads]).astype(
            dtype=np.float32)

        sess = Parallelized_Session(slammer_id, shm, infer_input_feed_info, infer_output_feed_info)
        for _ in range(10):
            sess.run(None, {"inputs": dummy_inputs,
                            "input_state": dummy_state,
                            "input_state_matrix": dummy_state_matrix})
        s = time.time()
        for i in tqdm(range(1000)):
            sess.run(None, {"inputs": dummy_inputs,
                            "input_state": dummy_state,
                            "input_state_matrix": dummy_state_matrix})
        its = 1000 / (time.time() - s)
        print("Slammer:", slammer_id, "did:", its, "slams per second!")


    slammers = []
    for slammer_id in range(num_workers):
        slammer = mp.Process(target=slam, args=(slammer_id, infer_input_feed_info, infer_output_feed_info, shms[slammer_id]))
        slammer.start()

    server = mp.Process(target=start_server, args=(batched_inputs_feed_info, batched_outputs_feed_info, shms, providers, "Gomoku/Cache/model_ctx.onnx"))
    server.start()
    # start_server(batched_inputs_feed_info, batched_outputs_feed_info, shms, providers, "Gomoku/Cache/model_ctx.onnx")

    for slammer in slammers:
        slammer.join()

    # server.terminate()
