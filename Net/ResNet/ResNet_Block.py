import tensorflow as tf

# in reference to https://zhangtemplar.github.io/resnet/
class ResNet_Identity2D(tf.keras.layers.Layer):
    def __init__(self, filters=256, kernel_size=(3, 3), activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization(dtype="float32")

        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization(dtype="float32")

        self.activation_fn = tf.keras.layers.Activation(activation)

    def call(self, inputs):
        x = self.bn1(inputs)
        x = tf.keras.layers.Activation("relu")(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.activation_fn(x)
        x = self.conv2(x)
        x += inputs
        return x

class ResNet_Conv2D(tf.keras.layers.Layer):
    def __init__(self, filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)
        self.bn1 = tf.keras.layers.BatchNormalization(dtype="float32")

        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization(dtype="float32")

        self.residual_conv = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), strides=strides, padding=padding)

        self.activation_fn = tf.keras.layers.Activation(activation)

    def call(self, inputs):
        x = self.bn1(inputs)
        x = tf.keras.layers.Activation("relu")(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.activation_fn(x)
        x = self.conv2(x)
        x += self.residual_conv(inputs)
        return x
