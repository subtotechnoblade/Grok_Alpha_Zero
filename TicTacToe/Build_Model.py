import tensorflow as tf

def build_model_infer(input_shape, policy_shape, build_config):
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
    x = Batched_Net_Infer.Batch(tf.keras.layers.Conv2D(64, (5, 5), padding="same"))(x)
    # x = Batched_Net_Infer.Batch(tf.keras.layers.Conv2D(256, (3, 3), padding="same"))(x)
    # x = tf.keras.layers.Activation("relu")(x)
    # x = Batched_Net_Infer.Batch(tf.keras.layers.Conv2D(256, (3, 3), padding="same"))(x)
    # x = Batched_Net_Infer.Batch(tf.keras.layers.Conv2D(64, (3, 3), padding="same"))(x)
    # x = Batched_Net_Infer.Batch(tf.keras.layers.Conv2D(64, (3, 3), padding="same"))(x)
    for _ in range(2):
        x = Batched_Net_Infer.Batch(ResNet_Conv2D(64, (3, 3), activation="relu"))(x)
        x = Batched_Net_Infer.Batch(ResNet_Identity2D(64, (3, 3), activation="relu"))(x)
    # x = Batched_Net_Infer.Batch(tf.keras.layers.Conv2D(32, (3, 3), padding="same"))(x)
    # x = tf.keras.layers.Reshape((input_shape[0] * input_shape[1] * 64,))(x)
    # x = tf.keras.layers.Reshape((-1,))(x)

    # x = Batched_Net_Infer.Batch(tf.keras.layers.Dense(embed_size * 2))(x)
    # x = tf.keras.layers.Activation("relu")(x)
    # x = Batched_Net_Infer.Batch(tf.keras.layers.Dense(embed_size))(x) # (batch, game_length, 32)

    for layer_id in range(num_layers):
        x, state, state_matrix = RWKV_v6_Infer.RWKV_Block(layer_id, num_heads, embed_size, token_shift_hidden_dim, hidden_size)(x, state, state_matrix)

    policy = Batched_Net_Infer.Batch(tf.keras.layers.Conv2D(2, (1, 1), padding="valid"))(x)
    policy = tf.keras.layers.Reshape((-1,))(policy)
    policy = Batched_Net_Infer.Batch(tf.keras.layers.Dense(embed_size * 2))(policy)
    policy = tf.keras.layers.Activation("relu")(policy)
    policy = Batched_Net_Infer.Batch(tf.keras.layers.Dense(embed_size // 2))(policy)

    policy = Batched_Net_Infer.Batch(tf.keras.layers.Dense(policy_shape[0]))(policy)

    if build_config["use_stable_max"]:
        policy = Stablemax(name="policy")(policy) # MUST NAME THIS "policy"
    else:
        policy = tf.keras.layers.Activation("softmax", name="policy")(policy)  # MUST NAME THIS "policy"

    value = Batched_Net_Infer.Batch(tf.keras.layers.Conv2D(2, (1, 1), padding="valid"))(x)
    value = tf.keras.layers.Reshape((-1,))(value)

    value = Batched_Net_Infer.Batch(tf.keras.layers.Dense(embed_size * 2))(value)
    value = tf.keras.layers.Activation("relu")(value)
    value = Batched_Net_Infer.Batch(tf.keras.layers.Dense(embed_size // 2))(value)
    value = Batched_Net_Infer.Batch(tf.keras.layers.Dense(1))(value)
    value = tf.keras.layers.Activation("tanh", name="value")(value) # MUST NAME THIS "value"

    output_state, output_state_matrix = tf.keras.layers.Identity(name="output_state")(state), tf.keras.layers.Identity(name="output_state_matrix")(state_matrix)
    # Must include this as it is necessary to name the outputs


    # feel free to also use return tf.keras.Model(inputs=inputs, outputs=[policy, value, state, state_matrix])
    # Grok fast model most likey improves convergence
    return tf.keras.Model(inputs=inputs, outputs=[policy, value, output_state, output_state_matrix])
