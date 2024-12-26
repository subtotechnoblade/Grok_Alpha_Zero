import tensorflow as tf

class Batch_Dense(tf.keras.layers.Layer):
    def __init__(self, units, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(units, use_bias=use_bias)

    def call(self, inputs):
        return tf.map_fn(self.dense, inputs)


class Batch_Conv1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(1, 1), strides=(1, 1), **kwargs):
        super().__init__(**kwargs)
        self.conv1D= tf.keras.layers.Conv1D(filters, kernel_size, strides, use_bias=use_bias)

    def call(self, inputs):
        return tf.map_fn(self.conv1D, inputs)

class Batch_Conv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(1, 1), strides=(1, 1), **kwargs):
        super().__init__(**kwargs)
        self.conv2D= tf.keras.layers.Conv2D(filters, kernel_size, strides, use_bias=use_bias)

    def call(self, inputs):
        return tf.map_fn(self.conv2D, inputs)
