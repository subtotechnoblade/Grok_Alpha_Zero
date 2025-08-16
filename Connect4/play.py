import onnxruntime as rt
from Connect4 import Connect4
from MCTS import MCTS
providers = [
    ('TensorrtExecutionProvider', {
        # "trt_engine_cache_enable": True,
        # "trt_dump_ep_context_model": True,
        # "trt_builder_optimization_level": 5,
        # "trt_auxiliary_streams": 0,
        # "trt_ep_context_file_path": "Gomoku/Cache/",
        #
        # "trt_profile_min_shapes": f"inputs:1x15x15,input_state:{num_layers}x2x1x{embed_size},input_state_matrix:{num_layers}x1x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",
        # "trt_profile_max_shapes": f"inputs:{max_shape}x15x15,input_state:{num_layers}x2x{max_shape}x{embed_size},input_state_matrix:{num_layers}x{max_shape}x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",
        # "trt_profile_opt_shapes": f"inputs:{opt_shape}x15x15,input_state:{num_layers}x2x{opt_shape}x{embed_size},input_state_matrix:{num_layers}x{opt_shape}x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",
    }),
    # 'CUDAExecutionProvider',
    'CPUExecutionProvider'
]
session = rt.InferenceSession("Grok_Zero_Train/7/TRT_cache/model_ctx.onnx", providers=providers)


game = Connect4()
print(game.board)
mcts = MCTS(game, session,
            c_puct_init=1.25,
            tau=0.0)

human_first_player = False

winner = -2
while winner == -2:
    if human_first_player and game.get_next_player() == -1:
        action = game.input_action()
    else:
        action, probs = mcts.run(5000)
        print(action, probs)
    game.do_action(action)

    print(game.board)

    winner = game.check_win()

    mcts.prune_tree(action)




