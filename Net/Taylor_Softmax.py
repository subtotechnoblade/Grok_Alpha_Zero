import tensorflow as tf


class Taylor_Softmax(tf.keras.layers.Layer):
    def taylor_e(self, x):
        return (1 + x + x ** 2) / 2

    def call(self, inputs):
        batch_min = tf.expand_dims(tf.reduce_min(inputs, -1), -1)
        taylor_exp = self.taylor_e(inputs - batch_min)
        return taylor_exp / tf.expand_dims(tf.reduce_sum(taylor_exp, -1), -1)

if __name__ == "__main__":
    import numpy as np

    # x = np.array([[1, 2, 3], [2, 3, 4]], dtype=np.float32)
    x = np.ones((2, 9), dtype=np.float32)
    fn = Taylor_Softmax()
    print(fn(x))