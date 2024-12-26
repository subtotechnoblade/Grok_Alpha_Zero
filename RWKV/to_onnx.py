import tensorflow as tf
import onnxoptimizer
import tf2onnx
import onnx

def convert_RWKV_to_onnx(tf_model, input_signature, file_path):
    # Use from_function for tf functions
    # similar to [tf.TensorSpec((None, *gf.SHAPE[1:]), TF_DTYPE, name="x")]
    onnx_model, _ = tf2onnx.convert.from_keras(tf_model, input_signature)
    onnx.save(onnx_model, "test_test.onnx")
    onnx_model = onnxoptimizer.optimize(onnx_model, passes=['nop', 'eliminate_nop_cast', 'eliminate_nop_dropout', 'eliminate_nop_flatten',
                                    'extract_constant_to_initializer', 'eliminate_if_with_const_cond',
                                    'eliminate_nop_monotone_argmax', 'eliminate_nop_pad', 'eliminate_nop_concat',
                                    'eliminate_nop_split', 'eliminate_nop_expand', 'eliminate_shape_gather',
                                    'eliminate_slice_after_shape', 'eliminate_nop_transpose',
                                    'fuse_add_bias_into_conv', 'fuse_bn_into_conv', 'fuse_consecutive_concats',
                                    'fuse_consecutive_log_softmax', 'fuse_consecutive_reduce_unsqueeze',
                                    'fuse_consecutive_squeezes', 'fuse_consecutive_transposes',
                                    'fuse_matmul_add_bias_into_gemm', 'fuse_pad_into_conv', 'fuse_pad_into_pool',
                                    'fuse_transpose_into_gemm', 'fuse_concat_into_reshape', 'eliminate_nop_reshape',
                                    'eliminate_nop_with_unit', 'eliminate_common_subexpression', 'fuse_qkv',
                                    'fuse_consecutive_unsqueezes', 'eliminate_deadend', 'eliminate_identity',
                                    'eliminate_shape_op', 'fuse_consecutive_slices', 'eliminate_unused_initializer',
                                    'eliminate_duplicate_initializer'])
    print("optimized")
    # for output in onnx_model.graph.output:
    #     if len(output.type.tensor_type.shape.dim) == 2:
    #         output.name = "output"
    #     elif len(output.type.tensor_type.shape.dim) == 3:
    #         output.name = "output_state"
    #     elif len(output.type.tensor_type.shape.dim) == 4:
    #         output.name = "output_state_matrix"
    #     else:
    #         raise ValueError("This output doesn't exist")
    #     print(output)
    onnx.save(onnx_model, file_path)

if __name__ == "__main__":
    import time
    import numpy as np
    from RWKV_v6_infer import make_test_model_infer

    embed_size, batch_size, num_heads, num_layers = 64, 1, 8, 5

    # model = make_test_model_infer(embed_size, num_heads, num_layers)
    # model.load_weights("test_model.weights.h5")
    # input_signature = [tf.TensorSpec((None, embed_size), tf.float32, name="inputs"),
    #                    tf.TensorSpec((num_layers, 2, None, embed_size), tf.float32, name="input_state"),
    #                    tf.TensorSpec((num_layers, None, num_heads, embed_size // num_heads, embed_size // num_heads), tf.float32, name="input_state_matrix"),
    #                    ]
    # convert_RWKV_to_onnx(model, input_signature, "test_model.onnx")

    import onnxruntime as rt
    providers = [
        ('TensorrtExecutionProvider', {
        "trt_engine_cache_enable":True,
        "trt_dump_ep_context_model": True,
        "trt_ep_context_file_path": "cache/"
        }),
        'CUDAExecutionProvider',
        'CPUExecutionProvider']

    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    s = time.time()
    # sess = rt.InferenceSession("test_model.onnx", sess_options=sess_options, providers=providers)
    # sess = rt.InferenceSession("test_test.onnx", sess_options=sess_options, providers=providers)
    sess = rt.InferenceSession("cache/test_model_ctx.onnx", providers=providers)
    print(time.time() - s)
    def create_states():
        return np.zeros((num_layers, 2, batch_size, embed_size), dtype=np.float32), np.zeros(
            (num_layers, batch_size, num_heads, embed_size // num_heads, embed_size // num_heads), np.float32)
    input_state, input_state_matrix = create_states()
    input_state_onnx, input_state_matrix_onnx = create_states()

    dummy_data = np.random.uniform(low=0, high=1, size=(100, batch_size, embed_size)).astype(np.float32)

    t_total = 0
    output1 = []
    output2 = []
    for data_point in dummy_data:
        # print(model_infer([data_point,input_state, input_state_matrix]))
        # x, input_state, input_state_matrix = [thing for thing in model([data_point,input_state, input_state_matrix]).values()]
        # note that RWKV_infer gives a dictionary which is necessary for onnxruntime

        # input_state = input_state.numpy()
        # input_state_matrix = input_state_matrix.numpy()
        # output1.append(x)

        s = time.time()
        results = sess.run(["rwkv__block_4_2", "rwkv__block_4", "rwkv__block_4_1"], {"inputs": data_point,
                                                                               "input_state": input_state_onnx,
                                                                               "input_state_matrix": input_state_matrix_onnx})
        dt = time.time() - s
        t_total += dt
        # print("Onnx took:", dt)

        input_state_onnx = results[1]
        input_state_matrix_onnx = results[2]
        output2.append(results[0])

    # output1 = np.array(output1)

    output2 = np.array(output2)
    print("Took",t_total,"seconds for", dummy_data.shape[0] * dummy_data.shape[1], "iterations")
    print((dummy_data.shape[1] * dummy_data.shape[0]) / t_total, "iterations per second")

    # print(np.allclose(output1, output2, atol=1e-5))
    # print(np.sum(np.abs(output1 - output2)))
#

