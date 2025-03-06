import tensorflow as tf

from Net.ResNet.ResNet_Block import ResNet_Identity2D, ResNet_Conv2D

from Net.Stablemax import Stablemax

from Net.Grok_Model import Grok_Fast_EMA_Model, Ortho_Model, Ortho_Grok_Fast_EMA_Model


def build_model(input_shape, policy_shape, build_config, train_config):
    # Since this is just and example for Gomoku
    # feel free to copy and modify

    embed_size = build_config["embed_size"]
    num_resnet_layers = build_config["num_resnet_layers"]
    num_filters = build_config["num_filters"]
    # input shape should be (3, 3)
    inputs = tf.keras.layers.Input(batch_shape=(None, *input_shape), name="inputs"),  # the name must be "inputs"
    x = inputs

    # reshaped_inputs = tf.keras.layers.Reshape((*input_shape, 1))(x)

    eyes = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
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
        x =  ResNet_Conv2D(int(num_filters * mul), (3, 3), strides=strides, padding=padding)(x)
        x =  ResNet_Identity2D(int(num_filters * mul), (3, 3))(x)

    policy = tf.keras.layers.Conv2D(4, (3, 3), padding="same")(x)
    policy = tf.keras.layers.Reshape((policy.shape[-3] * policy.shape[-2] * policy.shape[-1],))(policy)

    policy = tf.keras.layers.Dense(512)(policy)
    policy = tf.keras.layers.Activation("relu")(policy)
    policy = tf.keras.layers.Dense(policy_shape[0])(policy)

    if build_config["use_stable_max"]:
        policy = Stablemax(name="policy")(policy) # MUST NAME THIS "policy"
    else:
        policy = tf.keras.layers.Activation("softmax", name="policy")(policy)  # MUST NAME THIS "policy"

    value = tf.keras.layers.Conv2D(2, (2, 2), padding="same")(x)

    # value = Batched_Net_Infer.Batch(tf.keras.layers.GlobalAveragePooling2D())(value)
    value = tf.keras.layers.Reshape((value.shape[-3] * value.shape[-2] * value.shape[-1],))(value)
    value = tf.keras.layers.Dense(embed_size)(value)
    value = tf.keras.layers.Activation("relu")(value)
    value = tf.keras.layers.Dense(1)(value)  # MUST NAME THIS "value"
    value = tf.keras.layers.Activation("tanh", name="value")(value)

    # Must include this as it is necessary to name the outputs
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
