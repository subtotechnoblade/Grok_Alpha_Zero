import tensorflow as tf

# These are to be in replacement of the classes from Batched_Net
# To create a inference model as training is parallelized
# Ask Brian, if you are confused

class Batch_Dense(tf.keras.layers.Layer):
    def __init__(self, units, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(units, use_bias=use_bias)

    def call(self, inputs):
        return self.dense(inputs)

class Batch_Conv1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(1, 1), strides=(1, 1), use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.conv1D= tf.keras.layers.Conv1D(filters, kernel_size, strides, use_bias=use_bias)
    @tf.function
    def call(self, inputs):
        return self.conv1D(inputs)

class Batch_Conv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(1, 1), strides=(1, 1), padding="valid", use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.conv2D= tf.keras.layers.Conv2D(filters, kernel_size, strides, padding=padding, use_bias=use_bias)
    @tf.function
    def call(self, inputs):
        return self.conv2D(inputs)

class Batch_Conv3D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="valid", use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.conv3D= tf.keras.layers.Conv3D(filters, kernel_size, strides, padding=padding, use_bias=use_bias)
    @tf.function
    def call(self, inputs):
        return self.conv3D(inputs)