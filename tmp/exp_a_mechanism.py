"""Experiment A: what the reshape bug does to a single update step.

Takes REAL gradients from the real Connect4 model on real self play data, then
compares the update the buggy Muon would apply against the fixed one.
No training, so this runs in seconds.

    python tmp/exp_a_mechanism.py
"""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["NUMBA_DISABLE_JIT"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

from Connect4.Connect4 import Connect4, build_config, train_config, optimizer_config
from Connect4.Build_Model import build_model
from Net.Optimizers.muon import Muon
from Net.Alpha_Loss import Policy_Loss, Value_Loss
from Dataloader import Create_Dataset
from tmp.muon_buggy import BuggyMuon


def flat_cos(a, b):
    a, b = tf.reshape(a, [-1]), tf.reshape(b, [-1])
    return float(tf.tensordot(a, b, 1) / (tf.norm(a) * tf.norm(b) + 1e-30))


def effective_rank(matrix):
    """exp(entropy of the normalised singular value spectrum) - how many directions
    the update actually spans, as opposed to the hard rank bound."""
    s = tf.linalg.svd(matrix, compute_uv=False).numpy()
    s = s[s > 1e-9]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def main():
    keras_seed = 0
    tf.keras.utils.set_random_seed(keras_seed)

    game = Connect4()
    configs = (build_config, train_config, optimizer_config)
    model = build_model(game.get_input_state().shape, game.policy_shape, configs)

    train_dl, _ = Create_Dataset(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "Connect4", "Grok_Zero_Train"),
        num_previous_generations=1, train_batch_size=256, test_batch_size=256,
        train_percent=0.05, train_decay=0.2,
        test_percent=0.02, test_decay=0.03)
    states, (policies, values) = train_dl[0]
    policy_loss, value_loss = Policy_Loss(dtype="float32"), Value_Loss(dtype="float32")

    # The output Dense layers use kernel_initializer='zeros', so on the very first
    # backward pass the chain rule hands the ENTIRE trunk a gradient of exactly zero.
    # Warm up with plain Adam first so the gradients we compare on are representative.
    warmup = tf.keras.optimizers.Adam(1e-3)
    for step in range(30):
        s, (p, v) = train_dl[step % len(train_dl)]
        with tf.GradientTape() as tape:
            pp, vv = model(s, training=True)
            l = policy_loss(p, pp) + value_loss(v, vv)
        warmup.apply_gradients(zip(tape.gradient(l, model.trainable_variables),
                                   model.trainable_variables))
    print(f"warmed up 30 steps, loss now {float(l):.4f}")

    with tf.GradientTape() as tape:
        p_pred, v_pred = model(states, training=True)
        loss = policy_loss(policies, p_pred) + value_loss(values, v_pred)
    grads = tape.gradient(loss, model.trainable_variables)
    print(f"real batch of {len(states)} positions, loss {float(loss):.4f}\n")

    fixed, buggy = Muon(**{**optimizer_config["kwargs"], "learning_rate": 1e-2}), None
    buggy = BuggyMuon(**{**optimizer_config["kwargs"], "learning_rate": 1e-2})
    fixed.build(model.trainable_variables)
    buggy.build(model.trainable_variables)

    print("=" * 108)
    print(f"{'conv kernel':<22} {'NS matrix':>14} {'eff.rank':>9} | {'NS matrix':>14} {'eff.rank':>9} | "
          f"{'cos(u,-g)':>10} {'cos(u,-g)':>10}")
    print(f"{'':<22} {'BUGGY':>14} {'':>9} | {'FIXED':>14} {'':>9} | {'buggy':>10} {'fixed':>10}")
    print("-" * 108)

    for var, grad in zip(model.trainable_variables, grads):
        if grad is None or len(var.shape) != 4 or fixed._should_use_adamw(var):
            continue
        s = tuple(var.shape)
        ub = buggy.muon_update(var, grad, tf.constant(1e-2))
        uf = fixed.muon_update(var, grad, tf.constant(1e-2))

        # how the update looks when viewed the CORRECT way: (out_channels, everything else)
        as_matrix = lambda u: tf.reshape(tf.transpose(u, [3, 0, 1, 2]), (s[3], -1))
        buggy_mat_shape = (s[0], s[1] * s[2] * s[3])
        fixed_mat_shape = (s[3], s[0] * s[1] * s[2])

        print(f"{str(s):<22} {str(buggy_mat_shape):>14} {effective_rank(as_matrix(ub)):>9.1f} | "
              f"{str(fixed_mat_shape):>14} {effective_rank(as_matrix(uf)):>9.1f} | "
              f"{flat_cos(ub, -grad):>10.4f} {flat_cos(uf, -grad):>10.4f}")

    print("=" * 108)
    print("\nper output-channel share of the update's energy (should be even, Muon's whole point):")
    for var, grad in zip(model.trainable_variables, grads):
        if grad is None or len(var.shape) != 4 or fixed._should_use_adamw(var):
            continue
        s = tuple(var.shape)
        out = {}
        for name, u in (("buggy", buggy.muon_update(var, grad, tf.constant(1e-2))),
                        ("fixed", fixed.muon_update(var, grad, tf.constant(1e-2)))):
            per_ch = tf.norm(tf.reshape(tf.transpose(u, [3, 0, 1, 2]), (s[3], -1)), axis=1).numpy()
            out[name] = per_ch / (per_ch.mean() + 1e-30)
        print(f"  {str(s):<22} buggy: min {out['buggy'].min():.3f} max {out['buggy'].max():.3f} "
              f"cv {out['buggy'].std():.3f}   |   fixed: min {out['fixed'].min():.3f} "
              f"max {out['fixed'].max():.3f} cv {out['fixed'].std():.3f}")

    print("\nupdate magnitude actually applied (Frobenius norm, lr=1e-2):")
    for var, grad in zip(model.trainable_variables, grads):
        if grad is None or len(var.shape) != 4 or fixed._should_use_adamw(var):
            continue
        nb = float(tf.norm(buggy.muon_update(var, grad, tf.constant(1e-2))))
        nf = float(tf.norm(fixed.muon_update(var, grad, tf.constant(1e-2))))
        ratio = f"{nf / nb:.2f}x" if nb > 0 else "n/a"
        print(f"  {str(tuple(var.shape)):<22} buggy {nb:.5f}   fixed {nf:.5f}   ratio {ratio}")


if __name__ == "__main__":
    main()
