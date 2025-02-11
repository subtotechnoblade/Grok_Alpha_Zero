import tensorflow as tf

class Stablemax(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def s(self, x, epsilon=1e-30):
        return tf.where(x >= 0.0, x + 1.0, tf.math.divide_no_nan(1.0,  1.0 - x + epsilon))
    def call(self, inputs, axis=-1):
        s_x = self.s(inputs)

        return s_x / tf.reduce_sum(s_x, axis, keepdims=True)

class Stablemax_Binary_Crossentropy(tf.keras.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        # batch, timesteps, distribution
        loss = -y_true * tf.math.log(y_pred) - (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        return loss

class Stablemax_Binary_Focal_Crossentropy(tf.keras.Loss):
    def __init__(self, alpha=1.0, gamma=2.0, **kwargs):
        super().__init__(**kwargs)

        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        # batch, timesteps, distribution
        y_pred_converse= 1.0 - y_pred

        positives = -self.alpha * y_true * (y_pred_converse ** self.gamma) * tf.math.log(y_pred)
        negatives = -(1.0 - y_true) * (y_pred ** self.gamma) * tf.math.log(y_pred_converse)
        loss = positives + negatives
        return loss