"""Muon exactly as it was BEFORE the conv reshape fix.

Subclasses the real Muon and overrides only muon_update, so the two differ in
nothing except the reshape and the scaling factor. Everything else - momentum,
Newton-Schulz, weight decay, the AdamW path, layer routing - is inherited.

The bug: Keller's reference impl is PyTorch, where a conv kernel is stored as
(out_channels, in_channels, kh, kw), so `reshape(G, (G.shape[0], -1))` gives
(out_channels, everything_else). Keras stores kernels as (kh, kw, in_ch, out_ch),
so the same line gives (kernel_height, everything_else) = (3, N).
"""
import tensorflow as tf
from keras.src import ops

from Net.Optimizers.muon import Muon


class BuggyMuon(Muon):
    def muon_update(self, variable, gradient, learning_rate):
        muon_beta = ops.cast(self.muon_beta, variable.dtype)

        muon_m = self.muon_momentums[self._get_variable_index(variable)]
        muon_m_update = muon_beta * muon_m + gradient * (1 - muon_beta)
        muon_m.assign(muon_m_update)

        update = gradient + muon_m_update * muon_beta if self.nesterov else muon_m_update

        is_reshaped = False
        if update.ndim > 2:
            shape = ops.shape(update)
            update = ops.reshape(update, (shape[0], -1))   # <-- the bug
            is_reshaped = True

        update = self.zeropower_via_newtonschulz5(update)

        grad_shape = tf.shape(gradient)
        d_in = ops.cast(grad_shape[-1], dtype=tf.float32)    # <-- out_channels in keras
        d_out = ops.cast(grad_shape[-2], dtype=tf.float32)   # <-- in_channels in keras
        scaling_factor = 0.2 * ops.sqrt(ops.maximum(1.0, d_out / d_in))
        update *= scaling_factor

        if is_reshaped:
            update = tf.reshape(update, shape)

        return update * learning_rate
