from keras.src import ops
from keras.src.optimizers import optimizer
import tensorflow as tf

class Orthograd(optimizer.Optimizer):
    def __init__(self, base_optimizer, **kwargs):
        super().__init__(learning_rate=1.0, name="Orthograd_" + base_optimizer.name, **kwargs)
        self.base_optimizer = base_optimizer

    def build(self, variables):
        super().build(variables)
        self.base_optimizer.build(variables)

    def orthogonalize(self, var, grad):
        g = tf.reshape(grad, (-1,))
        w = tf.reshape(var, (-1,))

        w_norm_sq = ops.tensordot(w, w, axes=1) + 1e-30
        proj = ops.tensordot(w, g, axes=1) / w_norm_sq
        g_ortho = g - proj * w

        g_norm = tf.norm(g, ord=2)
        g_ortho_norm = tf.norm(g_ortho, 2) + 1e-30
        g_ortho_norm_scaled = g_ortho * (g_norm / g_ortho_norm)

        return tf.reshape(g_ortho_norm_scaled, grad.shape)

    def update_step(self, gradient, variable, learning_rate):
        if gradient is None:
            return

        # Only THIS optimizer's apply() runs, so the wrapped optimizer's step counter
        # never ticks. Without syncing it, the base optimizer's Adam bias correction is
        # frozen at t = 1 forever (1 - beta_1^t = 0.1, 1 - beta_2^t = 0.005).
        self.base_optimizer.iterations.assign(self.iterations)

        ortho_gradients = self.orthogonalize(variable, gradient)
        self.base_optimizer.update_step(ortho_gradients, variable, self.base_optimizer.learning_rate)

    def get_config(self):
        config = super().get_config()
        config.update({
            "base_optimizer": self.base_optimizer,
        })
        return config
