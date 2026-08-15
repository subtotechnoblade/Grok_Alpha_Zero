import math

import tensorflow as tf


class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup over warmup_frac of the run, then cosine to peak_lr * floor_frac."""

    def __init__(self, peak_lr, total_steps, warmup_frac=0.05, floor_frac=0.05,
                 name="WarmupCosine"):
        super().__init__()
        self.peak_lr = float(peak_lr)
        self.total_steps = max(int(total_steps), 1)
        self.warmup_frac = float(warmup_frac)
        self.floor_frac = float(floor_frac)
        self.name = name
        self.warmup_steps = max(int(self.total_steps * self.warmup_frac), 1)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm = self.peak_lr * (step + 1.0) / self.warmup_steps
        prog = tf.clip_by_value(
            (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1),
            0.0, 1.0)
        cos = self.floor_frac + (1.0 - self.floor_frac) * 0.5 * (1.0 + tf.cos(math.pi * prog))
        return tf.where(step < self.warmup_steps, warm, self.peak_lr * cos)

    def get_config(self):
        return {"peak_lr": self.peak_lr, "total_steps": self.total_steps,
                "warmup_frac": self.warmup_frac, "floor_frac": self.floor_frac,
                "name": self.name}


def step_decay_lr(generation, peak_lr, drop=10.0, every=40, floor_lr=None):
    """Divide peak_lr by `drop` every `every` generations."""
    lr = peak_lr / (drop ** (generation // every))
    return lr if floor_lr is None else max(lr, floor_lr)
