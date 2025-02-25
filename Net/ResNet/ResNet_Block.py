import tensorflow as tf

class ResNet2D(tf.keras.layers.Layer):
    def __init__(self, filters=256, kernel_size=(3, 3), activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.activation_fn = tf.keras.layers.Activation(activation)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = tf.keras.layers.Activation("relu")(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x += inputs
        x = self.activation_fn(x)
        return x
