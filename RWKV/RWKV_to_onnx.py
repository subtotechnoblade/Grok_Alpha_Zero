import tensorflow as tf
import tf2onnx
import onnx

def convert_RWKV_to_onnx(tf_model, input_signature, file_path):
    # input_signature = [tf.TensorSpec([3, 3], tf.float32, name='x')]
    # Use from_function for tf functions
    # similar to [tf.TensorSpec((None, *gf.SHAPE[1:]), TF_DTYPE, name="x")]
    onnx_model, _ = tf2onnx.convert.from_keras(tf_model, input_signature)
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

    embed_size, num_heads, num_layers = 64, 2, 1
    #
    model = make_test_model_infer(embed_size, num_heads, num_layers)
    model.load_weights("test_model.weights.h5")
    # input_signature = [tf.TensorSpec((1, embed_size), tf.float32, name="inputs"),
    #                    tf.TensorSpec((num_layers, 2, embed_size), tf.float32, name="input_state"),
    #                    tf.TensorSpec((num_layers, num_heads, embed_size // num_heads, embed_size // num_heads), tf.float32, name="input_state_matrix"),
    #                    ]
    # convert_RWKV_to_onnx(model, input_signature, "test_model.onnx")

    import onnxruntime as rt

    sess = rt.InferenceSession("test_model.onnx", providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])

    def create_states():
        return np.zeros((num_layers, 2, embed_size), dtype=np.float32), np.zeros(
            (num_layers, num_heads, embed_size // num_heads, embed_size // num_heads), np.float32)
    input_state, input_state_matrix = create_states()
    input_state_onnx, input_state_matrix_onnx = create_states()

    dummy_data = np.random.uniform(low=0, high=100, size=(100, embed_size)).astype(np.float32)

    t_total = 0
    output1 = []
    output2 = []
    for data_point in dummy_data:
        data_point = np.expand_dims(data_point, 0)
        # print(model_infer([data_point,input_state, input_state_matrix]))
        x, input_state, input_state_matrix = [thing for thing in model([data_point,input_state, input_state_matrix]).values()]
        # note that RWKV_infer gives a dictionary which is necessary for onnxruntime
        input_state = input_state.numpy()
        input_state_matrix = input_state_matrix.numpy()
        output1.append(x)

        s = time.time()
        results = sess.run(["rwkv__block_2", "rwkv__block", "rwkv__block_1"], {"inputs": data_point,
                                                                               "input_state": input_state_onnx,
                                                                               "input_state_matrix": input_state_matrix_onnx})
        dt = time.time() - s
        t_total += dt
        print("Onnx took:", dt)

        input_state_onnx = results[1]
        input_state_matrix_onnx = results[2]
        output2.append(results[0])
    output1 = np.array(output1)
    output2 = np.array(output2)
    print("Took",t_total,"seconds for", dummy_data.shape[0], "iterations")
    print(dummy_data.shape[0] / t_total, "iterations per second")

    print(np.allclose(output1, output2, atol=1e-5))
    print(np.sum(np.abs(output1 - output2)))



