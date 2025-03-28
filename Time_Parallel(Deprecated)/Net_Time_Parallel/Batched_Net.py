import tensorflow as tf

# This is the parallelized wrapper for common layers such as Dense and Conv
# These are parallelized over the batch and number of moves dim
# For example id the input is (batch_size, num_moves, board_shape) assuming board shape is a matrix
# we have to matrix multiply num_moves meany times over the board matrix
# But we also have to do all of that over the batch dimension
# Thus these use the map_fn to repeat the process again
# Ask Brian if you are confused
# Just wrap this class over Dense, Conv layers
class Batch(tf.keras.layers.Layer):
    def __init__(self, operation, **kwargs):
        super().__init__(**kwargs)
        self.operation = operation

    def call(self, inputs):
        return tf.map_fn(self.operation, inputs, parallel_iterations=64)
        # combined = tf.reshape(inputs, tf.concat([[-1], tf.shape(inputs)[2:]], 0))
        # y = self.operation(combined)
        # return tf.reshape(y, tf.concat([tf.shape(inputs)[:2], tf.shape(y)[1:]], 0))