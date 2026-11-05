import tensorflow as tf


def value_output(value, train_config):
    """Last value layer: tanh "value", or a win/draw/loss softmax "wdl" when wdl_head is on."""
    if train_config.get("wdl_head", False):
        value = tf.keras.layers.Dense(3, dtype="float32", name="value_3")(value)
        return tf.keras.layers.Activation("softmax", dtype="float32", name="wdl")(value)
    value = tf.keras.layers.Dense(1, dtype="float32", name="value_3")(value)  # MUST NAME THIS "value"
    return tf.keras.layers.Activation("tanh", dtype="float32", name="value")(value)


def extra_heads(trunk, policy_features, policy_channels, policy_size, train_config, bn_momentum, num_aux_heads=None):
    """Training-only outputs, in this order: soft_policy, opponent_policy, aux. None of them are exported."""
    outputs = []
    if train_config.get("soft_policy", False):
        soft = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(trunk)
        soft = tf.keras.layers.Activation("relu")(soft)
        soft = tf.keras.layers.Conv2D(32, (3, 3), padding="same", kernel_initializer='he_normal')(soft)
        soft = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(soft)
        soft = tf.keras.layers.Activation("relu")(soft)
        soft = tf.keras.layers.Conv2D(policy_channels, (1, 1), padding="same", dtype="float32")(soft)
        soft = tf.keras.layers.Reshape((policy_size,), dtype="float32")(soft)
        outputs.append(tf.keras.layers.Activation("softmax", dtype="float32", name="soft_policy")(soft))

    if train_config.get("opponent_policy", False):
        opponent = tf.keras.layers.Conv2D(policy_channels, (1, 1), padding="same", dtype="float32")(policy_features)
        opponent = tf.keras.layers.Reshape((policy_size,), dtype="float32")(opponent)
        outputs.append(tf.keras.layers.Activation("softmax", dtype="float32", name="opponent_policy")(opponent))

    if train_config.get("aux_heads", False):
        if not num_aux_heads:
            raise ValueError("aux_heads is on but this game defines no auxiliary targets")
        aux = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(trunk)
        aux = tf.keras.layers.Activation("relu")(aux)
        aux = tf.keras.layers.Conv2D(32, (1, 1), padding="same", kernel_initializer='he_normal')(aux)
        aux = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(aux)
        aux = tf.keras.layers.Activation("relu")(aux)
        aux = tf.keras.layers.Concatenate()([tf.keras.layers.GlobalAveragePooling2D()(aux),
                                             tf.keras.layers.GlobalMaxPooling2D()(aux)])
        aux = tf.keras.layers.Dense(128, kernel_initializer='he_normal')(aux)
        aux = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(aux)
        aux = tf.keras.layers.Activation("relu")(aux)
        outputs.append(tf.keras.layers.Dense(num_aux_heads, dtype="float32", name="aux")(aux))
    return outputs
