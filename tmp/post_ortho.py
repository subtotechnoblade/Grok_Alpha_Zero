"""OrthoGrad applied to the UPDATE instead of the gradient.

The stock Orthograd projects the gradient, then hands it to the base optimizer, whose
transform (Adam's 1/sqrt(v), Muon's Newton-Schulz) puts a radial component straight back.
Under BatchNorm the gradient is already orthogonal to W - measured cos = 0.00000 for every
conv that feeds a BN - so the stock version removes nothing and the growth it was meant to
stop happens downstream of it.

This projects the update the base optimizer actually applied, which is the quantity that
moves W. The base optimizer writes into the variable, so we snapshot first and recover the
update by difference.

NOTE: a purely radial update is exactly what decoupled weight decay is, so this cancels
weight decay by construction. Keep weight_decay small or zero when using it.
"""
import tensorflow as tf
from keras.src import ops
from keras.src.optimizers import optimizer


class PostOrthograd(optimizer.Optimizer):
    def __init__(self, base_optimizer, **kwargs):
        super().__init__(learning_rate=1.0, name="PostOrthograd_" + base_optimizer.name, **kwargs)
        self.base_optimizer = base_optimizer

    def build(self, variables):
        super().build(variables)
        self.base_optimizer.build(variables)

    def orthogonalize(self, var, update):
        u = tf.reshape(update, (-1,))
        w = tf.reshape(var, (-1,))
        proj = ops.tensordot(w, u, axes=1) / (ops.tensordot(w, w, axes=1) + 1e-30)
        u_ortho = u - proj * w
        # keep the step length the base optimizer chose, same convention as Orthograd
        u_ortho = u_ortho * (tf.norm(u, ord=2) / (tf.norm(u_ortho, ord=2) + 1e-30))
        return tf.reshape(u_ortho, update.shape)

    def update_step(self, gradient, variable, learning_rate):
        if gradient is None:
            return
        self.base_optimizer.iterations.assign(self.iterations)

        before = ops.convert_to_tensor(variable)
        self.base_optimizer.update_step(gradient, variable, self.base_optimizer.learning_rate)
        applied = before - variable                      # what the base optimizer just did
        variable.assign(before - self.orthogonalize(before, applied))

    def get_config(self):
        config = super().get_config()
        config.update({"base_optimizer": self.base_optimizer})
        return config
