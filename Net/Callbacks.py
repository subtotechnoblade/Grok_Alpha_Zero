import tensorflow as tf


class NormalizedValLoss(tf.keras.callbacks.Callback):
    """logs['val_norm'] = sum of each val loss divided by its own first-epoch value.

    Must be listed before ModelCheckpoint in callbacks.
    """

    def __init__(self, keys=("val_value_loss", "val_policy_loss")):
        super().__init__()
        self.keys = tuple(keys)
        self.first = {}

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        total = 0.0
        for key in self.keys:
            value = logs.get(key)
            if value is None:
                return
            self.first.setdefault(key, value if value > 0 else 1.0)
            total += value / self.first[key]
        logs["val_norm"] = total
