import tensorflow as tf

from Net.ResNet.ResNet_Block import ResNet_Block, Global_Pool_Block

from Net.Stablemax import Stablemax

from Net.Grok_Model import Grok_Fast_EMA_Model, Ortho_Model, Ortho_Grok_Fast_EMA_Model

from Net.Heads import value_output, extra_heads
from Net.Inputs import network_input
from Quoridor.game import NUM_AUX_HEADS


def build_model(input_shape, policy_shape, configs):
    build_config, train_config, optimizer_config = configs

    num_resnet_layers = build_config["num_resnet_layers"]
    num_filters = build_config["num_filters"]
    bn_momentum = build_config.get("bn_momentum", 0.99)
    # input shape should be (3, 3)
    inputs, x = network_input(input_shape, train_config)

    eyes = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', name="eyes")(x)
    eyes = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(eyes)
    eyes = tf.keras.layers.Activation("relu")(eyes)

    x = eyes
    mul = 1
    global_pool_blocks = build_config.get("global_pool_blocks", ())
    for layer_id in range(num_resnet_layers):
        if layer_id in global_pool_blocks:
            x = Global_Pool_Block(int(num_filters * mul), build_config.get("pool_filters", 32), (3, 3),
                                  bn_momentum=bn_momentum)(x)
        else:
            x = ResNet_Block(int(num_filters * mul), (3, 3), strides=(1, 1), padding="same", bn_momentum=bn_momentum)(x)

    # pre activation
    policy = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(x)
    policy = tf.keras.layers.Activation("relu")(policy)
    policy = tf.keras.layers.Conv2D(32, (3, 3), padding="same", kernel_initializer='he_normal')(policy)

    policy = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(policy)
    policy_features = tf.keras.layers.Activation("relu")(policy)
    policy = tf.keras.layers.Conv2D(3, (1, 1), padding="same", dtype="float32", name="policy_2")(policy_features)  # pawn, H wall, V wall per square
    policy = tf.keras.layers.Reshape((policy_shape[0],), dtype="float32")(policy)  # NOTE THAT THIS IS A LOGIT not prob

    if train_config["use_gumbel"]:
        policy = tf.keras.layers.Activation("linear", dtype="float32", name="policy")(policy)
    else:
        if build_config["use_stablemax"]:
            policy = Stablemax(name="policy", dtype="float32")(policy)  # MUST NAME THIS "policy"
        else:
            policy = tf.keras.layers.Activation("softmax", dtype="float32", name="policy")(policy)  # MUST NAME THIS "policy"

    value = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(x)
    value = tf.keras.layers.Activation("relu")(value)
    value = tf.keras.layers.Conv2D(32, (1, 1), padding="same", kernel_initializer='he_normal', name="value_1")(value)

    value = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(value)
    value = tf.keras.layers.Activation("relu")(value)
    value = tf.keras.layers.Concatenate()([tf.keras.layers.GlobalAveragePooling2D()(value),
                                           tf.keras.layers.GlobalMaxPooling2D()(value)])
    value = tf.keras.layers.Dense(128, kernel_initializer='he_normal', name="value_2")(value)

    value = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(value)
    value = tf.keras.layers.Activation("relu")(value)
    value = value_output(value, train_config)

    outputs = [policy, value] + extra_heads(x, policy_features, 3, policy_shape[0], train_config, bn_momentum,
                                            num_aux_heads=NUM_AUX_HEADS)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    from Quoridor.game import Quoridor
    from Quoridor.config import build_config, train_config, optimizer_config
    game = Quoridor()
    model = build_model(game.get_input_state().shape, game.policy_shape, (build_config, train_config, optimizer_config))
    model.summary()