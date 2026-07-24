from keras.src import ops
from keras.src.optimizers import optimizer
import tensorflow as tf

class Grokfast_EMA(optimizer.Optimizer):
    def __init__(self, base_optimizer, alpha=0.99, lamb=5.0, **kwargs):
        super().__init__(learning_rate=1.0, name="Grokfast_EMA_" +base_optimizer.name, **kwargs)
        self.base_optimizer = base_optimizer
        self.alpha = alpha
        self.lamb = lamb

    def build(self, variables):
        super().build(variables)
        self.base_optimizer.build(variables)
        self.ema_grads = {self._var_key(var): tf.Variable(tf.zeros_like(var), trainable=False) for var in variables}

    def filter_gradient(self, ema_grad:tf.Variable, grad):
        ema_grad.assign(ema_grad.value() * self.alpha + grad * (1 - self.alpha))
        return grad + ema_grad.value() * self.lamb
    def update_step(self, gradient, variable, learning_rate):
        if gradient is None:
            return

        # Only THIS optimizer's apply() runs, so the wrapped optimizer's step counter
        # never ticks. Without syncing it, the base optimizer's Adam bias correction is
        # frozen at t = 1 forever (1 - beta_1^t = 0.1, 1 - beta_2^t = 0.005).
        self.base_optimizer.iterations.assign(self.iterations)

        ema_grad = self.ema_grads[self._var_key(variable)]
        filtered_grad = self.filter_gradient(ema_grad, gradient)
        self.base_optimizer.update_step(filtered_grad, variable, self.base_optimizer.learning_rate)

    def get_config(self):
        config = super().get_config()
        config.update({
            "base_optimizer": self.base_optimizer,
            "alpha": self.alpha,
            "lamb": self.lamb})
        return config
