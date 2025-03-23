import tensorflow as tf

from Net.Stablemax import Stablemax

from Net.ResNet.ResNet_Block import ResNet_Identity2D, ResNet_Conv2D
from Net.Grok_Model import Grok_Fast_EMA_Model, Ortho_Model, Ortho_Grok_Fast_EMA_Model

def build_model(input_shape, policy_shape, build_config, train_config):
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # Since this is just and example for Gomoku
    # feel free to copy and modify

    # input shape should be (3, 3)
    inputs = tf.keras.layers.Input(batch_shape=(None, *input_shape), name="inputs") # the name must be "inputs"

    x = tf.keras.layers.Conv2D(128, (5, 5), padding="same")(inputs)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(x)

    for _ in range(build_config["num_resnet_layers"]):
        x = ResNet_Conv2D(64, (3, 3), activation="relu")(x)
        x = ResNet_Identity2D(64, (3, 3), activation="relu")(x)

    # policy = tf.keras.layers.BatchNormalization()(x)
    policy = tf.keras.layers.Conv2D(2, (1, 1), padding="valid")(x)
    policy = tf.keras.layers.Reshape((-1,))(policy)
    policy = tf.keras.layers.Dense(128)(policy)
    policy = tf.keras.layers.Activation("relu")(policy)
    policy = tf.keras.layers.Dense(64)(policy)


    if train_config["use_gumbel"]:
        policy = tf.keras.layers.Dense(policy_shape[0], name="policy")(policy) # NOTE THAT THIS IS A LOGIT not prob
    else:
        policy = tf.keras.layers.Dense(policy_shape[0])(policy) # NOTE THAT THIS IS A LOGIT not prob
        if build_config["use_stablemax"]:
            policy = Stablemax(name="policy")(policy)  # MUST NAME THIS "policy"
        else:
            policy = tf.keras.layers.Activation("softmax", name="policy")(policy)  # MUST NAME THIS "policy"


    # value = tf.keras.layers.BatchNormalization()(x)
    value = tf.keras.layers.Conv2D(2, (1, 1), padding="valid")(x)
    value = tf.keras.layers.Reshape((-1,))(value)

    value = tf.keras.layers.Dense(128)(value)
    value = tf.keras.layers.Dense(64)(value)
    value = tf.keras.layers.Activation("relu")(value)
    value = tf.keras.layers.Dense(1)(value)
    value = tf.keras.layers.Activation("tanh", name="value")(value) # MUST NAME THIS "value"

    # Must include this as it is necessary to name the outputs


    # feel free to also use return tf.keras.Model(inputs=inputs, outputs=[policy, value, state, state_matrix])
    # Grok fast model most likey improves convergence
    if build_config["use_grok_fast"] and build_config["use_orthograd"]:
        return Ortho_Grok_Fast_EMA_Model(inputs=inputs,outputs=[policy, value],
                                         lamb=build_config["grok_lambda"],
                                         alpha=0.99)
    elif build_config["use_grok_fast"]:
        return Grok_Fast_EMA_Model(inputs=inputs, outputs=[policy, value],
                                   lamb=build_config["grok_lambda"], alpha=0.99)
    elif build_config["use_orthograd"]:
        return Ortho_Model(inputs=inputs, outputs=[policy, value])
    else:
        return tf.keras.Model(inputs=inputs, outputs=[policy, value])
