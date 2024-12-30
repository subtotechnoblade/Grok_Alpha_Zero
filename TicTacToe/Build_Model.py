import tensorflow as tf
# pip install tensorflow

from Net.RWKV import RWKV_v6, RWKV_v6_Infer
from Net import Batched_Net, Batched_Net_Infer
from Net.Grok_Model import Grok_Fast_EMA_Model
# Note the imports will not be the same as these ^, but import RWKV.RWKV_v6


# This is just a template
def build_model(input_shape, policy_shape, build_config):
    # Since this is just and example for Gomoku
    # feel free to copy and modify

    embed_size = build_config["embed_size"]
    num_heads = build_config["num_heads"]
    num_layers = build_config["num_layers"]
    token_shift_hidden_dim = build_config["token_shift_hidden_dim"]
    hidden_size = build_config["hidden_size"]

    # input shape should be (3, 3)
    inputs = tf.keras.layers.Input(batch_shape=(None, None, *input_shape), name="inputs")
    # x = inputs
    x = tf.keras.layers.Reshape((-1, input_shape[0] * input_shape[1]))(inputs)


    # feel free to also use return Grok_Fast_EMA_Model(inputs=inputs, outputs=[policy, value])
    # Grok fast model most likey improves convergence
    # return Grok_Fast_EMA_Model(inputs=inputs, outputs=[policy, value])

def build_model_infer(input_shape, policy_shape, build_config):
    # Since this is just and example for Gomoku
    # feel free to copy and modify

    embed_size = build_config["embed_size"]
    num_heads = build_config["num_heads"]
    num_layers = build_config["num_layers"]
    token_shift_hidden_dim = build_config["token_shift_hidden_dim"]
    hidden_size = build_config["hidden_size"]

    # input shape should be (3, 3)
    inputs = [tf.keras.layers.Input(batch_shape=(None, *input_shape), name="inputs"), # the name must be "inputs"
              tf.keras.layers.Input(batch_shape=(num_layers, 2, None, embed_size), name="input_state"), # the name must be "input_state"
              tf.keras.layers.Input(batch_shape=(num_layers, None, num_heads, embed_size // num_heads, embed_size // num_heads), name="input_state_matrix"),# the name must be "input_state_matrix"
              ]
    x, state, state_matrix = inputs

    output_state, output_state_matrix = tf.keras.layers.Identity(name="state_matrix")(state), tf.keras.layers.Identity(name="output_state_matrix")(state_matrix)
    # Must include this as it is necessary to name the outputs


    # feel free to also use return Grok_Fast_EMA_Model(inputs=inputs, outputs=[policy, value, state, state_matrix])
    # Grok fast model most likey improves convergence
    # return Grok_Fast_EMA_Model(inputs=inputs, outputs=[policy, value, output_state, output_state_matrix])



if __name__ == '__main__':
    import numpy as np
    from Gomoku.Gomoku import Gomoku, build_config
    # print(build_config)
    batch_size = 10
    # Testing code to verify that both the train and infer version of the model result in the same outputs
    game = Gomoku()
    model = build_model(game.get_state().shape, game.policy_shape, build_config)
    tf.keras.utils.plot_model(model, "model_diagram.png",
                              show_shapes=True,
                              show_layer_names=True,
                              expand_nested=True)
    model.save_weights("test_model.weights.h5")
    model.summary()
    dummy_data = np.random.randint(low=-1, high=2, size=(batch_size, 2, 15, 15))
    # 10 is the length of the game in moves, 15, 15 is the dim of the board

    policy1, value1 = model(dummy_data)
    policy1 = np.array(policy1)
    value1 = np.array(value1)
    print(policy1.shape, value1.shape)

    def create_states(build_config):
        embed_size =build_config["embed_size"]
        num_heads = build_config["num_heads"]
        return np.zeros((build_config["num_layers"], 2, batch_size, embed_size)), np.zeros(
            (build_config["num_layers"], batch_size, num_heads, embed_size // num_heads, embed_size // num_heads))


    # This is for the infer model
    model_infer = build_model_infer(game.get_state().shape, game.policy_shape, build_config)
    model_infer.load_weights("test_model.weights.h5")
    tf.keras.utils.plot_model(model_infer, "model_infer_diagram.png",
                              show_shapes=True,
                              show_layer_names=True,
                              expand_nested=True)

    state, state_matrix = create_states(build_config)
    dummy_data = np.transpose(dummy_data, [1, 0, 2, 3])
    policy2, value2 = [], []
    for data_point in dummy_data:
        p, v, state, state_matrix = model_infer([data_point, state, state_matrix]) # Note that this must be constructed as a list
        state, state_matrix = state.numpy(), state_matrix.numpy() # necessary as we tensorflow only allows either all numpy inputs
        # or all tensorflow tensor inputs thus, all the inputs must be converted to numpy arrays
        policy2.append(p)
        value2.append(v)

    policy2 = np.array(policy2).transpose([1, 0, 2])
    value2 = np.array(value2).transpose([1, 0, 2])

    print(policy2.shape, value2.shape)

    print(np.allclose(policy1, policy2, rtol=1e-1))
    print(np.sum(np.abs(policy2 - policy1)) / len(policy1)) # average deviation over the entire length of policy

    print(np.allclose(value2, value1,  rtol=1e-2))
    print(np.sum(np.abs(value2 - value1)))
