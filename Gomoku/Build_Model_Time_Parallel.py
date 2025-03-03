import tensorflow as tf

from Net.ResNet.ResNet_Block import ResNet_Identity2D, ResNet_Conv2D

from Net_Time_Parallel import Batched_Net_Infer, Batched_Net
from Net.Stablemax import Stablemax

from Net.Grok_Model import Grok_Fast_EMA_Model, Ortho_Model, Ortho_Grok_Fast_EMA_Model


# Note the imports will not be the same as these ^, but import RWKV.RWKV_v6


# This is just a template
def build_model(input_shape, policy_shape, build_config):
    # Since this is just and example for Gomoku
    # feel free to copy and modify

    embed_size = build_config["embed_size"]
    num_heads = build_config["num_heads"]
    num_layers = build_config["num_layers"]
    num_resnet_layers = build_config["num_resnet_layers"]
    token_shift_hidden_dim = build_config["token_shift_hidden_dim"]
    hidden_size = build_config["hidden_size"]
    num_filters = build_config["num_filters"]

    inputs = tf.keras.layers.Input(batch_shape=(None, None, *input_shape), name="inputs")

    # reshaped_inputs = tf.keras.layers.Reshape((-1, *input_shape, 1))(inputs)
    eyes = Batched_Net.Batch(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same"))(inputs)
    # eyes = tf.keras.layers.LayerNormalization()(eyes)
    # eyes = tf.keras.layers.Activation("relu")(eyes)
    x = eyes
    mul = 1
    for layer_id in range(num_resnet_layers):
        padding="same"
        strides=(1, 1)
        if layer_id == 1:
            # padding="valid"
            mul *= 1.25
            strides = (1, 1)
        x =  Batched_Net.Batch(ResNet_Conv2D(int(num_filters * mul), (3, 3), strides=strides, padding=padding))(x)
        x =  Batched_Net.Batch(ResNet_Identity2D(int(num_filters * mul), (3, 3)))(x)

    # for layer_id in range(num_layers):  # -2 because later we use two layers for the policy and value
    #     x = Train_net.RWKV_Block(layer_id, num_heads, embed_size, token_shift_hidden_dim, hidden_size)(x)


    policy = Batched_Net.Batch(tf.keras.layers.Conv2D(4, (3, 3), padding="same"))(x)

    policy = tf.keras.layers.Reshape((-1, policy.shape[-3] * policy.shape[-2] * policy.shape[-1]))(policy)

    policy = Batched_Net.Batch(tf.keras.layers.Dense(512))(policy)
    policy = tf.keras.layers.Activation("relu")(policy)
    policy = Batched_Net.Batch(tf.keras.layers.Dense(policy_shape[0]))(policy)

    if build_config["use_stable_max"]:
        policy = Stablemax(name="policy")(policy) # MUST NAME THIS "policy"
    else:
        policy = tf.keras.layers.Activation("softmax", name="policy")(policy) # MUST NAME THIS "policy"


    value = Batched_Net.Batch(tf.keras.layers.Conv2D(2, (2, 2), padding="same"))(x)

    # value = Batched_Net.Batch(tf.keras.layers.GlobalAveragePooling2D())(value)
    value = tf.keras.layers.Reshape((-1, value.shape[-3] * value.shape[-2] * value.shape[-1]))(value)
    value = Batched_Net.Batch(tf.keras.layers.Dense(embed_size))(value)
    value = tf.keras.layers.Activation("relu")(value)
    value = Batched_Net.Batch(tf.keras.layers.Dense(1))(value)  # MUST NAME THIS "value"
    value = tf.keras.layers.Activation("tanh", name="value")(value)

    if build_config["use_grok_fast"] and build_config["use_orthograd"]:
        return Ortho_Grok_Fast_EMA_Model(inputs=inputs,outputs=[policy, value],
                                         lamb=build_config["grok_lambda"],
                                         alpha=0.99)
    elif build_config["use_grok_fast"]:
        return Grok_Fast_EMA_Model(inputs=inputs, outputs=[policy, value],
                                   lamb=build_config["grok_lambda"], alpha=0.99)
    elif build_config["use_orthograd"]:
        return Ortho_Model(inputs=inputs, outputs=[policy, value],
                           lamb=build_config["grok_lambda"], alpha=0.99)
    else:
        return tf.keras.Model(inputs=inputs, outputs=[policy, value])


def build_model_infer(input_shape, policy_shape, build_config):
    # Since this is just and example for Gomoku
    # feel free to copy and modify

    embed_size = build_config["embed_size"]
    num_heads = build_config["num_heads"]
    num_layers = build_config["num_layers"]
    num_resnet_layers = build_config["num_resnet_layers"]
    token_shift_hidden_dim = build_config["token_shift_hidden_dim"]
    hidden_size = build_config["hidden_size"]
    num_filters = build_config["num_filters"]
    # input shape should be (3, 3)
    inputs = [tf.keras.layers.Input(batch_shape=(None, *input_shape), name="inputs"),  # the name must be "inputs"
              tf.keras.layers.Input(batch_shape=(num_layers, 2, None, embed_size), name="input_state"),
              # the name must be "input_state"
              tf.keras.layers.Input(
                  batch_shape=(num_layers, None, num_heads, embed_size // num_heads, embed_size // num_heads),
                  name="input_state_matrix"),  # the name must be "input_state_matrix"
              ]
    x, state, state_matrix = inputs

    # reshaped_inputs = tf.keras.layers.Reshape((*input_shape, 1))(x)

    eyes = Batched_Net_Infer.Batch(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same"))(x)
    # eyes = tf.keras.layers.LayerNormalization()(eyes)
    # eyes = tf.keras.layers.Activation("relu")(eyes)

    x = eyes
    mul = 1
    for layer_id in range(num_resnet_layers):
        strides = (1, 1)
        padding="same"
        if layer_id == 1:
            # padding="valid"
            strides = (1, 1)
            mul *= 1.25
        x =  Batched_Net_Infer.Batch(ResNet_Conv2D(int(num_filters * mul), (3, 3), strides=strides, padding=padding))(x)
        x =  Batched_Net_Infer.Batch(ResNet_Identity2D(int(num_filters * mul), (3, 3)))(x)

    # for layer_id in range(num_layers):  # -2 because later we use two layers for the policy and value
    #     x, state, state_matrix = Infer_net.RWKV_Block(layer_id, num_heads, embed_size, token_shift_hidden_dim,
    #                                                       hidden_size)(x, state, state_matrix)

    policy = Batched_Net_Infer.Batch(tf.keras.layers.Conv2D(4, (3, 3), padding="same"))(x)
    policy = tf.keras.layers.Reshape((policy.shape[-3] * policy.shape[-2] * policy.shape[-1],))(policy)

    policy = Batched_Net_Infer.Batch(tf.keras.layers.Dense(512))(policy)
    policy = tf.keras.layers.Activation("relu")(policy)
    policy = Batched_Net_Infer.Batch(tf.keras.layers.Dense(policy_shape[0]))(policy)

    if build_config["use_stable_max"]:
        policy = Stablemax(name="policy")(policy) # MUST NAME THIS "policy"
    else:
        policy = tf.keras.layers.Activation("softmax", name="policy")(policy)  # MUST NAME THIS "policy"

    value = Batched_Net_Infer.Batch(tf.keras.layers.Conv2D(2, (2, 2), padding="same"))(x)

    # value = Batched_Net_Infer.Batch(tf.keras.layers.GlobalAveragePooling2D())(value)
    value = tf.keras.layers.Reshape((value.shape[-3] * value.shape[-2] * value.shape[-1],))(value)
    value = Batched_Net_Infer.Batch(tf.keras.layers.Dense(embed_size))(value)
    value = tf.keras.layers.Activation("relu")(value)
    value = Batched_Net_Infer.Batch(tf.keras.layers.Dense(1))(value)  # MUST NAME THIS "value"
    value = tf.keras.layers.Activation("tanh", name="value")(value)

    output_state, output_state_matrix = tf.keras.layers.Identity(name="output_state")(state), tf.keras.layers.Identity(
        name="output_state_matrix")(state_matrix)
    # Must include this as it is necessary to name the outputs

    return tf.keras.Model(inputs=inputs, outputs=[policy, value, output_state, output_state_matrix])


if __name__ == '__main__':
    import numpy as np
    from Gomoku import Gomoku, build_config

    # print(build_config)
    batch_size = 1
    # Testing code to verify that both the train and infer version of the model result in the same outputs
    game = Gomoku()
    model = build_model(game.get_input_state().shape, game.policy_shape, build_config)
    model.summary()
    # raise ValueError
    # raise ValueError
    # tf.keras.utils.plot_model(model, "model_diagram.png",
    #                           show_shapes=True,
    #                           show_layer_names=True,
    #                           expand_nested=True
    #                           )
    model.save_weights("test_model.weights.h5")
    # raise ValueError
    # model.save_weights("test_model.weights.h5")

    # model = tf.keras.models.load_model("test_model.weights.h5")
    # model.summary()

    dummy_data = np.random.randint(low=-1, high=2, size=(batch_size, 2, *game.get_input_state().shape))
    # 10 is the length of the game in moves, 15, 15 is the dim of the board

    policy1, value1 = model(dummy_data)
    policy1 = np.array(policy1)
    value1 = np.array(value1)
    print(policy1.shape, value1.shape)


    def create_states(build_config):
        embed_size = build_config["embed_size"]
        num_heads = build_config["num_heads"]
        return np.zeros((build_config["num_layers"], 2, batch_size, embed_size)), np.zeros(
            (build_config["num_layers"], batch_size, num_heads, embed_size // num_heads, embed_size // num_heads))


    # This is for the infer model
    model_infer = build_model_infer(game.get_input_state().shape, game.policy_shape, build_config)
    model_infer.load_weights("test_model.weights.h5")
    # tf.keras.utils.plot_model(model_infer, "model_infer_diagram.png",
    #                           show_shapes=True,
    #                           show_layer_names=True,
    #                           expand_nested=True)

    state, state_matrix = create_states(build_config)
    dummy_data = np.transpose(dummy_data, [1, 0, 2, 3, 4])
    policy2, value2 = [], []
    for data_point in dummy_data:
        p, v, state, state_matrix = model_infer(
            [data_point, state, state_matrix])  # Note that this must be constructed as a list
        state, state_matrix = state.numpy(), state_matrix.numpy()  # necessary as we tensorflow only allows either all numpy inputs
        # or all tensorflow tensor inputs thus, all the inputs must be converted to numpy arrays
        policy2.append(p)
        value2.append(v)

    policy2 = np.array(policy2).transpose([1, 0, 2])
    value2 = np.array(value2).transpose([1, 0, 2])

    print(policy2.shape, value2.shape)

    print(np.allclose(policy1, policy2, rtol=1e-1, atol=1e-2))
    print(np.sum(np.abs(policy2 - policy1)))  # average deviation over the entire length of policy

    print(np.allclose(value2, value1, rtol=1e-2))
    print(np.sum(np.abs(value2 - value1)))

