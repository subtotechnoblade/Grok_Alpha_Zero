import tensorflow as tf

# in reference to https://zhangtemplar.github.io/resnet/
class ResNet_Identity2D(tf.keras.layers.Layer):
    def __init__(self, filters=256, kernel_size=(3, 3), activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), padding="same")
        self.ln1 = tf.keras.layers.LayerNormalization()

        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), padding="same")
        self.ln2 = tf.keras.layers.LayerNormalization()

        self.activation_fn = tf.keras.layers.Activation(activation)

    def call(self, inputs):
        x = self.ln1(inputs)
        x = tf.keras.layers.Activation("relu")(x)
        x = self.conv1(x)

        x = self.ln2(x)
        x = self.activation_fn(x)
        x = self.conv2(x)
        x += inputs
        return x

class ResNet_Conv2D(tf.keras.layers.Layer):
    def __init__(self, filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)
        self.ln1 = tf.keras.layers.LayerNormalization()

        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), padding="same")
        self.ln2 = tf.keras.layers.LayerNormalization()

        self.residual_conv = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), strides=strides, padding=padding)

        self.activation_fn = tf.keras.layers.Activation(activation)

    def call(self, inputs):
        x = self.ln1(inputs)
        x = tf.keras.layers.Activation("relu")(x)
        x = self.conv1(x)

        x = self.ln2(x)
        x = self.activation_fn(x)
        x = self.conv2(x)
        x += self.residual_conv(inputs)
        return x
