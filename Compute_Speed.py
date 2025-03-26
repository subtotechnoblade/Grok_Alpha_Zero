import time
import numpy as np
def convert_shape(shape):
    assert len(shape) > 0
    str_shape = ""
    for dim in shape:
        str_shape += f"{dim}x"
    return str_shape[:-1]
def compute_speed(game_class,
              build_config,
              train_config,
              folder_path,
              iterations=500):
    import onnxruntime as rt # must do this for windows

    game = game_class()
    board_shape = game.board.shape
    str_board_shape = convert_shape(board_shape)

    onnx_file_path = f"{folder_path}/model.onnx"
    if train_config["use_gpu"]:
        if train_config["use_tensorrt"]:
            max_shape = train_config["num_workers"]
            onnx_file_path = f"{folder_path}/TRT_cache/model_ctx.onnx"
            providers = [
                ('TensorrtExecutionProvider', {
                    "trt_engine_cache_enable": True,
                    "trt_dump_ep_context_model": True,
                    "trt_builder_optimization_level": 5,
                    "trt_auxiliary_streams": 0,

                    "trt_ep_context_file_path": f"{folder_path}/TRT_cache/",
                    "trt_profile_min_shapes": f"inputs:1x{str_board_shape}",
                    "trt_profile_max_shapes": f"inputs:{max_shape}x{str_board_shape}",
                    "trt_profile_opt_shapes": f"inputs:{max_shape}x{str_board_shape}",
                }),
                'CUDAExecutionProvider',
                'CPUExecutionProvider']
        else:
            providers = [
                'CUDAExecutionProvider',
                'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']


    sess = rt.InferenceSession(onnx_file_path, providers=providers)

    dummy_inputs = np.random.uniform(-1, 2, (train_config["num_workers"], *game.get_input_state().shape)).astype(np.float32)

    for _ in range(10):
        policy, value = sess.run(["policy", "value"], input_feed={"inputs": dummy_inputs})

    s = time.time()
    for _ in range(iterations):
        policy, value = sess.run(["policy", "value"], input_feed={"inputs": dummy_inputs})
    time_taken = time.time() - s
    print(f"Took {time_taken / iterations} seconds per iteration at {iterations / time_taken:0.2f} it/s!")

if __name__ == "__main__":
    from TicTacToe.Tictactoe import TicTacToe, build_config, train_config

    compute_speed(TicTacToe, build_config, train_config, "TicTacToe/Grok_Zero_Train/0")