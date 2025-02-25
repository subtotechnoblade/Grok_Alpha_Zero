import os
import tensorflow as tf
import onnxoptimizer
import tf2onnx
import onnx

def convert_to_onnx(tf_model, input_signature, file_path): # must call this function "convert_to_onnx"
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # Use from_function for tf functions
    # similar to [tf.TensorSpec((None, *gf.SHAPE[1:]), TF_DTYPE, name="x")]
    onnx_model, _ = tf2onnx.convert.from_keras(tf_model, input_signature)

    onnx_model = onnxoptimizer.optimize(onnx_model)
    # , passes=['nop', 'eliminate_nop_cast', 'eliminate_nop_dropout', 'eliminate_nop_flatten',
    #                                 'extract_constant_to_initializer', 'eliminate_if_with_const_cond',
    #                                 'eliminate_nop_monotone_argmax', 'eliminate_nop_pad', 'eliminate_nop_concat',
    #                                 'eliminate_nop_split', 'eliminate_nop_expand', 'eliminate_shape_gather',
    #                                 'eliminate_slice_after_shape', 'eliminate_nop_transpose',
    #                                 'fuse_add_bias_into_conv', 'fuse_bn_into_conv', 'fuse_consecutive_concats',
    #                                 'fuse_consecutive_log_softmax', 'fuse_consecutive_reduce_unsqueeze',
    #                                 'fuse_consecutive_squeezes', 'fuse_consecutive_transposes',
    #                                 'fuse_matmul_add_bias_into_gemm', 'fuse_pad_into_conv', 'fuse_pad_into_pool',
    #                                 'fuse_transpose_into_gemm', 'fuse_concat_into_reshape', 'eliminate_nop_reshape',
    #                                 'eliminate_nop_with_unit', 'eliminate_common_subexpression', 'fuse_qkv',
    #                                 'fuse_consecutive_unsqueezes', 'eliminate_deadend', 'eliminate_identity',
    #                                 'eliminate_shape_op', 'fuse_consecutive_slices', 'eliminate_unused_initializer',
    #                                 'eliminate_duplicate_initializer'])

    onnx.save(onnx_model, file_path)


if __name__ == "__main__":
    import time
    import numpy as np
    # from Gomoku.Build_Model import build_model, build_model_infer
    # from Gomoku.Gomoku import Gomoku, build_config

    from TicTacToe.Build_Model import build_model, build_model_infer
    from TicTacToe.Tictactoe import TicTacToe, build_config, train_config

    # Test and validation code for gomoku

    batch_size = 12
    embed_size, num_heads, num_layers = build_config["embed_size"], build_config["num_heads"], build_config["num_layers"]
    game = TicTacToe()

    model = build_model_infer(game.get_input_state().shape, game.policy_shape, build_config)
    generation = 0
    model.load_weights(f"TicTacToe/Grok_Zero_Train/{generation}/model.weights.h5")
    # input_signature = [tf.TensorSpec((None, 15, 15), tf.float32, name="inputs"),
    #                    tf.TensorSpec((num_layers, 2, None, embed_size), tf.float32, name="input_state"),
    #                    tf.TensorSpec((num_layers, None, num_heads, embed_size // num_heads, embed_size // num_heads), tf.float32, name="input_state_matrix"),
    #                    ]
    # convert_to_onnx(model, input_signature, "model.onnx")

    import onnxruntime as rt
    min_shape, max_shape, opt_shape = 1, 12, batch_size
    providers = [
        # ('TensorrtExecutionProvider', {
        # "trt_engine_cache_enable": True,
        # "trt_dump_ep_context_model": True,
        # "trt_builder_optimization_level": 5,
        # "trt_auxiliary_streams": 0,
        # "trt_ep_context_file_path": "Cache/",
        #
        # "trt_profile_min_shapes": f"inputs:{min_shape}x15x15,input_state:{num_layers}x2x{min_shape}x{embed_size},input_state_matrix:{num_layers}x{min_shape}x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",
        # "trt_profile_max_shapes": f"inputs:{max_shape}x15x15,input_state:{num_layers}x2x{max_shape}x{embed_size},input_state_matrix:{num_layers}x{max_shape}x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",
        # "trt_profile_opt_shapes": f"inputs:{opt_shape}x15x15,input_state:{num_layers}x2x{opt_shape}x{embed_size},input_state_matrix:{num_layers}x{opt_shape}x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",
        # }),
        # 'CUDAExecutionProvider',
        'CPUExecutionProvider']

    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

    # sess = rt.InferenceSession("test_test.onnx", sess_options=sess_options, providers=providers)
    sess = rt.InferenceSession(f"TicTacToe/Grok_Zero_Train/{generation}/model.onnx", sess_options=sess_options, providers=providers)
    # sess = rt.InferenceSession("Cache/model_ctx.onnx", sess_options=sess_options, providers=providers)
    # sess = rt.InferenceSession("cache/test_model_ctx.onnx", sess_options=sess_options, providers=providers)

    def create_states():
        return np.zeros((num_layers, 2, batch_size, embed_size), dtype=np.float32), np.zeros(
            (num_layers, batch_size, num_heads, embed_size // num_heads, embed_size // num_heads), np.float32)
    input_state, input_state_matrix = create_states()
    input_state_onnx, input_state_matrix_onnx = create_states()

    dummy_data = np.random.uniform(low=-1, high=2, size=(9, batch_size, *game.get_input_state().shape)).astype(np.float32)

    # for mini_size in range(1, batch_size + 1):
    #     print(mini_size)
    #     data_point = np.random.uniform(low=-1, high=2, size=(mini_size, 15, 15)).astype(np.float32)
    #     sess.run(["policy", "value", "output_state", "output_state_matrix"],
    #              {"inputs": data_point,
    #               "input_state": input_state_onnx,
    #               "input_state_matrix": input_state_matrix_onnx})

    t_total = 0
    output1_p = []
    output1_v = []

    output2_p = []
    output2_v = []
    for data_point in dummy_data:

        p, v, input_state, input_state_matrix = model([data_point, input_state, input_state_matrix])
        # note that RWKV_infer gives a dictionary which is necessary for onnxruntime

        input_state = input_state.numpy()
        input_state_matrix = input_state_matrix.numpy()
        output1_p.append(p)
        output1_v.append(v)

        s = time.time()
        policy, value, state, state_matrix = sess.run(["policy", "value", "output_state", "output_state_matrix"],
                                                      {"inputs": data_point,
                                                                "input_state": input_state_onnx,
                                                                "input_state_matrix": input_state_matrix_onnx})
        dt = time.time() - s
        t_total += dt
        # print("Onnx took:", dt)
        # print(policy)
        # raise ValueError

        input_state_onnx = state
        input_state_matrix_onnx = state_matrix
        output2_p.append(policy)
        output2_v.append(value)

    output1_p = np.array(output1_p)

    output2_p = np.array(output2_p)
    print("Took",t_total,"seconds for", dummy_data.shape[0] * dummy_data.shape[1], "iterations")
    print((dummy_data.shape[1] * dummy_data.shape[0]) / t_total, "iterations per second")

    print(np.allclose(output1_p, output2_p, atol=1e-5))
    print(np.sum(np.abs(output1_p - output2_p)))
    #
    # output1_v = np.array(output1_v)
    # output2_v = np.array(output2_v)
    # print(np.allclose(output1_v, output2_v, atol=1e-3))
    # print(np.sum(np.abs(output1_v - output2_v)))
#

