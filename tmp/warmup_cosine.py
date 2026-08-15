"""Linear warmup into cosine decay, measured in STEPS.

Two separate decays are in play and they must not be confused:

  WITHIN a generation   this schedule. The value head peaks around epoch 4 at a fixed
                        1e-2 and then decays while the policy keeps improving, so no
                        single epoch is right for both. Annealing inside the generation
                        is what makes the LAST epoch the best one for both heads.

  ACROSS generations    the existing lr_schedule(generation) in Connect4/config.py,
                        which sets the PEAK this schedule warms up to. Unchanged here.

Warmup exists because Adam's v_hat is near zero for the first handful of steps, so the
effective step is largest exactly when the weights are least able to absorb it.

Keras evaluates a LearningRateSchedule per step inside BaseOptimizer.apply() and hands
the result to update_step(), so Adam and Muon both pick it up untouched. Orthograd does
NOT - it passes base_optimizer.learning_rate directly (ortho_grad.py:57), which is the
schedule object rather than its value at this step. Do not combine the two.
"""
import numpy as np
import tensorflow as tf


class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, total_steps, warmup_steps=200, floor_frac=0.02):
        self.peak_lr = float(peak_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.floor_frac = float(floor_frac)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm = self.peak_lr * (step + 1.0) / max(self.warmup_steps, 1)
        prog = tf.clip_by_value(
            (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1),
            0.0, 1.0)
        cos = self.floor_frac + (1.0 - self.floor_frac) * 0.5 * (1.0 + tf.cos(np.pi * prog))
        return tf.where(step < self.warmup_steps, warm, self.peak_lr * cos)

    def get_config(self):
        return {"peak_lr": self.peak_lr, "total_steps": self.total_steps,
                "warmup_steps": self.warmup_steps, "floor_frac": self.floor_frac}


if __name__ == "__main__":
    s = WarmupCosine(1e-2, total_steps=4320, warmup_steps=200)
    for st in (0, 100, 200, 432, 864, 1728, 2592, 3456, 4319):
        print(f"step {st:>5} (epoch {st/432:>5.2f})  lr {float(s(st)):.6f}")
