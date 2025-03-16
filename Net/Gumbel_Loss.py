import tensorflow as tf
from Net.Stablemax import Stablemax
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
class Policy_Loss_Gumbel(tf.keras.Loss):
    def __init__(self, loss_fn=tf.keras.losses.KLDivergence(), activation_fn=None, **kwargs):
        super().__init__(**kwargs)
        self.activation_fn = activation_fn
        self.loss_fn = loss_fn

    def call(self, y_true, y_pred):
        y_pred = self.activation_fn(y_pred)
        return self.loss_fn(y_true, y_pred)
class Value_Loss_Gumbel(tf.keras.Loss):
    def __init__(self, loss_fn=tf.keras.losses.MeanSquaredError(), **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn

    def call(self, y_true, y_pred):
        return self.loss_fn(y_true, y_pred)

class KLD_Gumbel(tf.keras.Loss):
    def __init__(self, activation_fn, **kwargs):
        super().__init__(**kwargs)
        self.activation_fn = activation_fn
        self.loss_fn = tf.keras.losses.KLDivergence()

    def call(self, y_true, y_pred):
        y_pred = self.activation_fn(y_pred)
        return self.loss_fn(y_true, y_pred)


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