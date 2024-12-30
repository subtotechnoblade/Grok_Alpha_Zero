import tensorflow as tf

# This is the parallelized version of common layers such as Dense and Conv
# These are parallelized over the batch and number of moves dim
# For example id the input is (batch_size, num_moves, board_shape) assuming board shape is a matrix
# we have to matrix multiply num_moves meany times over the board matrix
# But we also have to do all of that over the batch dimension
# Thus these use the map_fn to repeat the process again
# Ask Brian if you are confused
class Batch_Dense(tf.keras.layers.Layer):
    def __init__(self, units, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(units, use_bias=use_bias)

    def call(self, inputs):
        return tf.map_fn(self.dense, inputs)


class Batch_Conv1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(1, 1), strides=(1, 1), use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.conv1D= tf.keras.layers.Conv1D(filters, kernel_size, strides, use_bias=use_bias)

    def call(self, inputs):
        return tf.map_fn(self.conv1D, inputs)

class Batch_Conv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(1, 1), strides=(1, 1), use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.conv2D= tf.keras.layers.Conv2D(filters, kernel_size, strides, use_bias=use_bias)

    def call(self, inputs):
        return tf.map_fn(self.conv2D, inputs)


class Batch_Conv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(1, 1, 1), strides=(1, 1, 1), use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.conv3D= tf.keras.layers.Conv3D(filters, kernel_size, strides, use_bias=use_bias)

    def call(self, inputs):
        return tf.map_fn(self.conv3D, inputs)