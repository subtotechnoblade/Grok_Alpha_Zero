import tensorflow as tf

class Stablemax(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def s(self, x, epsilon=1e-30):
        return tf.where(x >= 0.0, x + 1.0, tf.math.divide_no_nan(1.0,  1.0 - x + epsilon))
    def call(self, inputs, axis=-1):
        s_x = self.s(inputs)

        return s_x / tf.reduce_sum(s_x, axis, keepdims=True)