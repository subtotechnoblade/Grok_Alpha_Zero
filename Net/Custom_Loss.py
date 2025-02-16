import tensorflow as tf
from Utils import Stablemax_Binary_Crossentropy

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