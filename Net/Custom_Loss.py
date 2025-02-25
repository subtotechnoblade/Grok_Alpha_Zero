import tensorflow as tf

class Stable_Categorical_Crossentropy(tf.keras.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        # batch, timesteps, distribution
        loss = -y_true * tf.math.log(y_pred)
        return loss

class Stable_Categorical_Focal_Crossentropy(tf.keras.Loss):
    def __init__(self, alpha=0.5, gamma=2.0, **kwargs):
        super().__init__(**kwargs)

        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        # batch, timesteps, distribution
        loss = -self.alpha * ((1.0 - y_pred) ** self.gamma) * y_true * tf.math.log(y_pred)
        return loss
class Policy_Loss(tf.keras.Loss):
    # Note that reduction MUST be None
    def __init__(self, loss_fn=tf.keras.losses.CategoricalFocalCrossentropy(reduction=None),**kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn
    def call(self, y_true, y_pred):
        # Note that y_true is a vector [batch_size, timestep, policy_shape]
        loss = self.loss_fn(y_true, y_pred)
        # returns (batch_size,) as the shape for BCE loss

        mask = tf.cast(y_true[:, :, 0] != -2.0, tf.float32) # if the first number of a policy is -2
        # we mask out that sample of loss, this is because games have difference number of moves
        # thus we have to pad some samples to fit the maximum length of a game
        return tf.reduce_mean(tf.reduce_sum(mask * loss, axis=-1) / tf.reduce_sum(mask, axis=-1))

class Value_Loss(tf.keras.Loss):
    # Note that reduction MUST be None
    def __init__(self, loss_fn=tf.keras.losses.MeanSquaredError(reduction=None), **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn
    def call(self, y_true, y_pred):
        # Note that y_true is a vector [batch_size, policy_shape]
        loss = self.loss_fn(y_true, y_pred)
        # returns (batch_size,) as the shape for BCE loss
        mask = tf.cast(y_true[:, :, 0] != -2.0, tf.float32) # if the first number of a policy is -1
        # we mask out that sample of loss, this is because games have difference number of moves
        # thus we have to pad some samples to fit the maximum length of a game
        return tf.reduce_mean(tf.reduce_sum(mask * loss, axis=-1) / tf.reduce_sum(mask, axis=-1))


class KLD(tf.keras.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = tf.keras.losses.KLDivergence(reduction=None)

    def call(self, y_true, y_pred):
        loss = self.loss_fn(y_true, y_pred)
        mask = tf.cast(y_true[:, :, 0] != -2.0, tf.float32)
        masked_loss = tf.reduce_mean(tf.reduce_sum(mask * loss, axis=-1) / tf.reduce_sum(mask, -1))
        return masked_loss

if __name__ == "__main__":
    from Stablemax import Stablemax
    import numpy as np
    fn1 = Stable_Categorical_Focal_Crossentropy(alpha=0.25, reduction=None)
    # fn1 = Stablemax_Categorical_Focal_Crossentropy(reduction=None)
    # fn2 = tf.keras.losses.CategoricalCrossentropy(reduction=None)
    fn2 = tf.keras.losses.CategoricalFocalCrossentropy(reduction=None)

    stable_max = Stablemax()
    target = stable_max(np.random.random((1, 2, 9)))
    pred = stable_max(np.random.random((1, 2, 9)))

    print(fn1(target, pred))
    print(tf.reduce_sum(fn1(target, pred), -1))
    print(fn2(target, pred))