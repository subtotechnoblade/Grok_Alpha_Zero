import tensorflow as tf

class Policy_Loss(tf.keras.Loss):
    # Note that reduction MUST be None
    def __init__(self, loss_fn=tf.losses.BinaryCrossentropy(reduction=None),**kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn
    def call(self, y_true, y_pred):
        # Note that y_true is a vector [batch_size, policy_shape]
        loss = self.loss_fn(y_true, y_pred)
        # returns (batch_size,) as the shape for BCE loss

        mask = tf.cast(y_true[:, 0] != -1, tf.float32) # if the first number of a policy is -1
        # we mask out that sample of loss, this is because games have difference number of moves
        # thus we have to pad some samples to fit the maximum length of a game
        masked_loss = tf.reduce_mean(mask * loss)
        return masked_loss

class Value_Loss(tf.keras.Loss):
    # Note that reduction MUST be None
    def __init__(self, loss_fn=tf.keras.losses.MSE(reduction=None), **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn
    def call(self, y_true, y_pred):
        # Note that y_true is a vector [batch_size, policy_shape]
        loss = self.loss_fn(y_true, y_pred)
        # returns (batch_size,) as the shape for BCE loss

        mask = tf.cast(y_true != -1, tf.float32) # if the first number of a policy is -1
        # we mask out that sample of loss, this is because games have difference number of moves
        # thus we have to pad some samples to fit the maximum length of a game
        masked_loss = tf.reduce_mean(mask * loss)
        return masked_loss

