import math

import numpy as np
import tensorflow as tf


class EMAWeights(tf.keras.callbacks.Callback):
    """Exponential moving average of the trainable weights. At each epoch end the averaged weights, with BatchNorm
    statistics recomputed on bn_batches training batches, are validated (val_* logs) and left in place so a
    following ModelCheckpoint saves them. The raw weights and statistics come back before the next epoch trains.

    Must be listed before NormalizedValLoss and ModelCheckpoint, and fit() gets no validation_data."""

    def __init__(self, decay, validation_data, train_data, bn_batches=50):
        super().__init__()
        self.decay = float(decay)
        self.validation_data = validation_data
        self.train_data = train_data
        self.bn_batches = bn_batches
        self.raw = None

    def on_train_begin(self, logs=None):
        self.shadow = [tf.Variable(v, trainable=False) for v in self.model.trainable_variables]

    @tf.function
    def _update(self):
        for s_var, var in zip(self.shadow, self.model.trainable_variables):
            s_var.assign(self.decay * s_var + (1.0 - self.decay) * var)

    def on_train_batch_end(self, batch, logs=None):
        self._update()

    def on_epoch_begin(self, epoch, logs=None):
        if self.raw is not None:
            for var, value in zip(self.model.trainable_variables + self.model.non_trainable_variables, self.raw):
                var.assign(value)
            self.raw = None

    def on_epoch_end(self, epoch, logs=None):
        self.raw = [v.numpy() for v in self.model.trainable_variables + self.model.non_trainable_variables]
        for var, s_var in zip(self.model.trainable_variables, self.shadow):
            var.assign(s_var)
        for i in np.random.permutation(len(self.train_data))[:self.bn_batches]:
            self.model(self.train_data[int(i)][0], training=True)
        val = self.model.evaluate(self.validation_data, verbose=0, return_dict=True)
        if logs is not None:
            logs.update({f"val_{k}": float(v) for k, v in val.items()})


class NormalizedValLoss(tf.keras.callbacks.Callback):
    """logs['val_norm'] = weighted sum of each val loss divided by its baseline value: the
    pre-training val loss when given and finite, else the first-epoch value.

    Must be listed before ModelCheckpoint in callbacks.
    """

    def __init__(self, keys=("val_value_loss", "val_policy_loss"), weights=None, baseline=None):
        super().__init__()
        self.keys = tuple(keys)
        self.weights = dict(weights or {})
        self.first = {k: v for k, v in (baseline or {}).items() if math.isfinite(v) and v > 0}

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        total = 0.0
        for key in self.keys:
            value = logs.get(key)
            if value is None:
                return
            self.first.setdefault(key, value if value > 0 else 1.0)
            total += self.weights.get(key, 1.0) * value / self.first[key]
        logs["val_norm"] = total
