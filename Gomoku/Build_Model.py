import tensorflow as tf

from Net.ResNet.ResNet_Block import ResNet_Block

from Net.Stablemax import Stablemax

from Net.Grok_Model import Grok_Fast_EMA_Model, Ortho_Model, Ortho_Grok_Fast_EMA_Model


def build_model(input_shape, policy_shape, build_config, train_config):
    # Since this is just and example for Gomoku
    # feel free to copy and modify

    num_resnet_layers = build_config["num_resnet_layers"]
    num_filters = build_config["num_filters"]
    # input shape should be (3, 3)
    inputs = tf.keras.layers.Input(batch_shape=(None, *input_shape), name="inputs")  # the name must be "inputs"
    x = inputs

    eyes = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal')(x)
    eyes = tf.keras.layers.BatchNormalization()(eyes)
    eyes = tf.keras.layers.Activation("relu")(eyes)

    x = eyes
    mul = 1
    for layer_id in range(num_resnet_layers):
        x = ResNet_Block(int(num_filters * mul), (3, 3), strides=(1, 1), padding="same")(x)

    # pre activation
    policy = tf.keras.layers.BatchNormalization()(x)
    policy = tf.keras.layers.Activation("relu")(policy)
    policy = tf.keras.layers.Conv2D(32, (3, 3), padding="same", kernel_initializer='he_normal')(policy)

    policy = tf.keras.layers.BatchNormalization()(policy)
    policy = tf.keras.layers.Activation("relu")(policy)
    policy = tf.keras.layers.Conv2D(8, (3, 3), padding="same", kernel_initializer='he_normal')(policy)

    policy = tf.keras.layers.Reshape((policy.shape[-3] * policy.shape[-2] * policy.shape[-1],))(policy)

    policy = tf.keras.layers.BatchNormalization()(policy)
    policy = tf.keras.layers.Activation("relu")(policy)
    policy = tf.keras.layers.Dense(512, kernel_initializer='he_normal')(policy)

    policy = tf.keras.layers.BatchNormalization()(policy)
    policy = tf.keras.layers.Activation("relu")(policy)

    # policy = tf.keras.layers.Dense(512)(policy)
    # policy = tf.keras.layers.BatchNormalization()(policy)
    # policy = tf.keras.layers.Activation("relu")(policy)

    policy = tf.keras.layers.Dense(policy_shape[0], dtype="float32")(policy) # NOTE THAT THIS IS A LOGIT not prob
    policy *= build_config["rr_alpha"] # rich representation

    if train_config["use_gumbel"]:
        policy = tf.keras.layers.Activation("linear", dtype="float32", name="policy")(policy)
    else:
        if build_config["use_stablemax"]:
            policy = Stablemax(name="policy", dtype="float32")(policy)  # MUST NAME THIS "policy"
        else:
            policy = tf.keras.layers.Activation("softmax", dtype="float64", name="policy")(policy)  # MUST NAME THIS "policy"

    value = tf.keras.layers.BatchNormalization()(x)
    value = tf.keras.layers.Activation("relu")(value)
    value = tf.keras.layers.Conv2D(32, (3, 3), padding="same", kernel_initializer='he_normal')(value)

    value = tf.keras.layers.BatchNormalization()(value)
    value = tf.keras.layers.Activation("relu")(value)
    value = tf.keras.layers.Conv2D(4, (1, 1), padding="same", kernel_initializer='he_normal')(value)

    value = tf.keras.layers.Reshape((value.shape[-3] * value.shape[-2] * value.shape[-1],))(value)

    value = tf.keras.layers.BatchNormalization()(value)
    value = tf.keras.layers.Activation("relu")(value)
    value = tf.keras.layers.Dense(256, kernel_initializer='he_normal')(value)

    value = tf.keras.layers.BatchNormalization()(value)
    value = tf.keras.layers.Activation("relu")(value)
    value = tf.keras.layers.Dense(128, kernel_initializer='he_normal')(value)

    value = tf.keras.layers.BatchNormalization()(value)
    value = tf.keras.layers.Activation("relu")(value)
    value = tf.keras.layers.Dense(1, dtype="float32")(value)  # MUST NAME THIS "value"
    value *= build_config["rr_alpha"] # rich representation

    value = tf.keras.layers.Activation("tanh", dtype="float32", name="value")(value)

    # Must include this as it is necessary to name the outputs
    if build_config["use_grok_fast"] and build_config["use_orthograd"]:
        return Ortho_Grok_Fast_EMA_Model(inputs=inputs,outputs=[policy, value],
                                         lamb=build_config["grok_fast_lambda"],
                                         alpha=0.99)
    elif build_config["use_grok_fast"]:
        return Grok_Fast_EMA_Model(inputs=inputs, outputs=[policy, value],
                                   lamb=build_config["grok_fast_lambda"], alpha=0.99)
    elif build_config["use_orthograd"]:
        return Ortho_Model(inputs=inputs, outputs=[policy, value])
    else:
        return tf.keras.Model(inputs=inputs, outputs=[policy, value])

if __name__ == "__main__":
    from Gomoku import Gomoku, build_config, train_config
    game = Gomoku()
    model = build_model(game.get_input_state().shape, game.policy_shape, build_config,train_config)
    model.summary()