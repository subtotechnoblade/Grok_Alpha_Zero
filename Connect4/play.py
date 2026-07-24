import onnxruntime as rt
from Connect4 import Connect4
from MCTS import MCTS
from MCTS_Gumbel import MCTS_Gumbel
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
# session = rt.InferenceSession("Grok_Zero_Train/3/TRT_cache/model_ctx.onnx", providers=providers)
session = rt.InferenceSession("Grok_Zero_Train/3/model.onnx", providers=providers)


game = Connect4()
print(game.board)
mcts = MCTS(game,
            session,
            # session=None,
            c_puct_init=3.5,
            use_dirichlet=False,
            use_njit=True,
            tau=0.0)
#
# mcts = MCTS_Gumbel(game, session,
#                    use_gumbel_noise=False,
#             m=7, c_scale=1.0)


human_first_player = 0
winner = -2
while winner == -2:
    if game.get_next_player() == (-1 if human_first_player else 1):
        action = game.input_action()
    else:
        action, probs = mcts.run(1000)
        print(action, probs)
    game.do_action(action)

    print(game.board)

    winner = game.check_win()

    mcts.prune_tree(action)


# import pygame as pg
#
#
# class Connect4_GUI:
#     def __init__(self):
#         self.game = Connect4()
#
#
#     def update(self):
#
#
#         pass
#     def render(self, events):
#
#
# def run()


