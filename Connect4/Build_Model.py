import tensorflow as tf

from Net.ResNet.ResNet_Block import ResNet_Block, Recurrent_ResNet

from Net.Stablemax import Stablemax

from Net.Grok_Model import Grok_Fast_EMA_Model, Ortho_Model, Ortho_Grok_Fast_EMA_Model


def build_model(input_shape, policy_shape, configs):
    build_config, train_config, optimizer_config = configs

    num_resnet_layers = build_config["num_resnet_layers"]
    num_filters = build_config["num_filters"]
    # input shape should be (3, 3)
    inputs = tf.keras.layers.Input(batch_shape=(None, *input_shape), name="inputs")  # the name must be "inputs"
    # x = tf.keras.layers.Reshape((6, 7, 1))(inputs)

    # reshaped_inputs = tf.keras.layers.Reshape((*input_shape, 1))(x)

    eyes = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_normal", name="eyes")(inputs)
    eyes = tf.keras.layers.BatchNormalization()(eyes)
    eyes = tf.keras.layers.Activation("relu")(eyes)

    x = eyes
    mul = 1
    for layer_id in range(num_resnet_layers):
        # strides = (1, 1)
        # padding="same"

        # if layer_id == 1:
        #     # padding="valid"
        #     strides = (1, 1)
        #     mul *= 1.25
        x = ResNet_Block(int(num_filters * mul), (3, 3), strides=(1, 1), padding="same")(x)
        # x = ResNet_Identity2D(int(num_filters * mul), (3, 3))(x) # replacing the ResNet_COnv2D
        # x = ResNet_Identity2D(int(num_filters * mul), (3, 3))(x)
    # x = Recurrent_ResNet(num_resnet_layers, num_filters)(x)

    policy = tf.keras.layers.BatchNormalization()(x)
    policy = tf.keras.layers.Activation("relu")(policy)
    policy = tf.keras.layers.Conv2D(8, (3, 3), padding="same", kernel_initializer="he_normal")(policy)
    policy = tf.keras.layers.Reshape((policy.shape[-3] * policy.shape[-2] * policy.shape[-1],))(policy)

    policy = tf.keras.layers.BatchNormalization()(policy)
    policy = tf.keras.layers.Activation("relu")(policy)

    policy = tf.keras.layers.Dense(128, kernel_initializer="he_normal")(policy)
    policy = tf.keras.layers.BatchNormalization()(policy)
    policy = tf.keras.layers.Activation("relu")(policy)

    policy = tf.keras.layers.Dense(64, kernel_initializer="he_normal")(policy)
    policy = tf.keras.layers.BatchNormalization()(policy)
    policy = tf.keras.layers.Activation("relu")(policy)

    policy = tf.keras.layers.Dense(policy_shape[0], kernel_initializer='zeros', bias_initializer='zeros', dtype="float32", name="policy_1")(policy) # NOTE THAT THIS IS A LOGIT not prob
    if train_config["use_gumbel"]:
        policy = tf.keras.layers.Activation("linear", dtype="float32", name="policy")(policy)
    else:
        if build_config["use_stablemax"]:
            policy = Stablemax(name="policy", dtype="float32")(policy)  # MUST NAME THIS "policy"
        else:
            policy = tf.keras.layers.Activation("softmax", dtype="float32", name="policy")(policy)  # MUST NAME THIS "policy"


    value = tf.keras.layers.BatchNormalization()(x)
    value = tf.keras.layers.Activation("relu")(value)
    value = tf.keras.layers.Conv2D(8, (3, 3), padding="same", kernel_initializer="he_normal")(value)

    # value = Batched_Net_Infer.Batch(tf.keras.layers.GlobalAveragePooling2D())(value)
    value = tf.keras.layers.Reshape((value.shape[-3] * value.shape[-2] * value.shape[-1],))(value)
    value = tf.keras.layers.BatchNormalization()(value)
    value = tf.keras.layers.Activation("relu")(value)

    value = tf.keras.layers.Dense(128, kernel_initializer="he_normal")(value)
    value = tf.keras.layers.BatchNormalization()(value)
    value = tf.keras.layers.Activation("relu")(value)

    value = tf.keras.layers.Dense(64, kernel_initializer="he_normal")(value)
    value = tf.keras.layers.BatchNormalization()(value)
    value = tf.keras.layers.Activation("relu")(value)

    value = tf.keras.layers.Dense(1, kernel_initializer='zeros', bias_initializer='zeros', dtype="float32", name="value_1")(value)  # MUST NAME THIS "value"
    value = tf.keras.layers.Activation("tanh", dtype="float32", name="value")(value)

    return tf.keras.Model(inputs=inputs, outputs=[policy, value])

if __name__ == "__main__":
    from Connect4.game import Connect4
    from Connect4.config import build_config, train_config
    game = Connect4()
    model = build_model(game.get_input_state().shape, game.policy_shape, build_config,train_config)
    model.summary()