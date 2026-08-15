import re

from keras.src import ops
from keras.src.optimizers import optimizer
import tensorflow as tf

class Orthograd(optimizer.Optimizer):
    """Removes the component of each gradient parallel to the weight it updates.

    exclude_layers: regex patterns matched against variable.path. Matching variables
        are passed through untouched. Empty by default.
    min_ndim: variables with fewer dimensions are passed through. The projection is
        identically zero for a 1-element variable.
    """

    def __init__(self, base_optimizer, exclude_layers=(), min_ndim=2, **kwargs):
        super().__init__(learning_rate=1.0, name="Orthograd_" + base_optimizer.name, **kwargs)
        self.base_optimizer = base_optimizer
        self.exclude_layers = list(exclude_layers)
        self.min_ndim = min_ndim

    def _should_orthogonalize(self, variable):
        if len(variable.shape) < self.min_ndim:
            return False
        return not any(re.search(p, variable.path) for p in self.exclude_layers)

    def build(self, variables):
        super().build(variables)
        self.base_optimizer.build(variables)
        self.ortho_mask = [self._should_orthogonalize(var) for var in variables]

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

        if self.ortho_mask[self._get_variable_index(variable)]:
            gradient = self.orthogonalize(variable, gradient)
        self.base_optimizer.update_step(gradient, variable, self.base_optimizer.learning_rate)

    def get_config(self):
        config = super().get_config()
        config.update({
            "base_optimizer": self.base_optimizer,
            "exclude_layers": self.exclude_layers,
            "min_ndim": self.min_ndim,
        })
        return config
