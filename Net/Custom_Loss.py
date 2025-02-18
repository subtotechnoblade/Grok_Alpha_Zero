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
class Policy_Loss(tf.keras.Loss):
    # Note that reduction MUST be None
    def __init__(self, loss_fn=Stablemax_Binary_Crossentropy(),**kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn
    def call(self, y_true, y_pred):
        # Note that y_true is a vector [batch_size, timestep, policy_shape]
        loss = self.loss_fn(y_true, y_pred)
        # returns (batch_size,) as the shape for BCE loss

        mask = tf.cast(y_true[:, :, 0] != -2, tf.float32) # if the first number of a policy is -2
        # we mask out that sample of loss, this is because games have difference number of moves
        # thus we have to pad some samples to fit the maximum length of a game
        masked_loss = tf.reduce_mean(mask * loss)
        return masked_loss

class Value_Loss(tf.keras.Loss):
    # Note that reduction MUST be None
    def __init__(self, loss_fn=tf.keras.losses.MeanSquaredError(reduction=None), **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn
    def call(self, y_true, y_pred):
        # Note that y_true is a vector [batch_size, policy_shape]
        loss = self.loss_fn(y_true, y_pred)
        # returns (batch_size,) as the shape for BCE loss

        mask = tf.cast(y_true != -2, tf.float32) # if the first number of a policy is -1
        # we mask out that sample of loss, this is because games have difference number of moves
        # thus we have to pad some samples to fit the maximum length of a game
        masked_loss = tf.reduce_mean(mask * loss)
        return masked_loss

if __name__ == "__main__":
    import numpy as np
    fn = Stablemax_Binary_Crossentropy(reduction=None)


    target = np.ones((2, 3, 3)) * 1
    pred = np.random.random((2, 3, 3))

    print(fn(target, pred))