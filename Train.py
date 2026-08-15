import os

import tensorflow as tf

from Net.Stablemax import Stablemax
from Net.Alpha_Loss import Policy_Loss, Value_Loss, KLD
from Net.Gumbel_Loss import Policy_Loss_Gumbel, Value_Loss_Gumbel, KLD_Gumbel

from Net.Optimizers.adam import Adam
from Net.Optimizers.nadam import Nadam
from Net.Optimizers.muon import Muon
from Net.Optimizers.ortho_grad import Orthograd
from Net.Optimizers.grok_fast import Grokfast_EMA
from Net.Schedules import WarmupCosine
from Net.Callbacks import NormalizedValLoss


def train(train_dataloader, test_dataloader, model, learning_rate, configs: list[dict],
          run_dir="Grok_Zero_Train"):
    build_config, train_config, optimizer_config = configs
    # assume that save_folder path is Grok_Zero_Train/current_generation + 1
    # run_dir is where tmp_best lives. pipeline.py passes the absolute
    # <repo>/<game>/Grok_Zero_Train so the scratch weights land next to the run
    # instead of wherever the process happened to be launched from
    tmp_best = f"{run_dir}/tmp_best.weights.h5"
    accum_steps = optimizer_config.get("gradient_accumulation_steps")
    if accum_steps is not None and accum_steps >= 2:
        learning_rate /= accum_steps  # this is to counter the gradients being scaled by accum_steps thus we divide

    optimizer = {"adam": Adam, "nadam": Nadam, "muon": Muon}[optimizer_config["optimizer"].lower()]
    kwargs = dict(optimizer_config["kwargs"])  # a copy: pipeline.py re-reads kwargs each generation
    if optimizer_config.get("use_orthograd", False):
        print("use_orthograd is on: step schedule disabled, using a flat lr for this generation.")
        kwargs["learning_rate"] = learning_rate
    else:
        total_steps = len(train_dataloader) * optimizer_config["train_epochs"]
        kwargs["learning_rate"] = WarmupCosine(
            learning_rate, total_steps,
            warmup_frac=optimizer_config.get("lr_warmup_frac", 0.05),
            floor_frac=optimizer_config.get("lr_floor_frac", 0.05))
    optimizer = optimizer(**kwargs)

    # order matters, grads -> orthograd -> grokfast EMA filter -> optimizer
    # orthograd(grokfast_EMA(base_optimizer))
    if optimizer_config.get("use_grokfast", False):
        optimizer = Grokfast_EMA(optimizer, lamb=optimizer_config["grokfast_lambda"])

    if optimizer_config.get("use_orthograd", False):
        optimizer = Orthograd(optimizer,
                              exclude_layers=optimizer_config.get("orthograd_exclude", ()))

    if not train_config.get("use_gumbel", False):
        policy_loss, value_loss, kld = Policy_Loss(dtype="float32"), Value_Loss(dtype="float32"), KLD(dtype="float32", reduction=None)
    else:
        if build_config.get("use_stablemax", False):
            activation_fn = Stablemax(dtype="float32")
        else:
            activation_fn = tf.keras.layers.Activation("softmax", dtype="float64")
            # activation_fn = Taylor_Softmax(name="softmax", dtype="float64")
        policy_loss, value_loss, kld = (Policy_Loss_Gumbel(activation_fn=activation_fn, dtype="float64"),
                                        Value_Loss_Gumbel(dtype="float64"),
                                        KLD_Gumbel(activation_fn=activation_fn, dtype="float64"))

    model.compile(optimizer=optimizer,
                  loss={"policy": policy_loss, "value": value_loss},
                  weighted_metrics={"policy": tf.keras.metrics.KLDivergence(name="policy_kld")}
                  )

    norm_val_loss = NormalizedValLoss()  # must precede the checkpoint, it writes val_norm

    save_best_model = tf.keras.callbacks.ModelCheckpoint(
        filepath=tmp_best,
        monitor='val_norm',
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        verbose=1
    )

    model.fit(train_dataloader,
              validation_data=test_dataloader,
              epochs=optimizer_config["train_epochs"],
              callbacks=[norm_val_loss, save_best_model])

    model.load_weights(tmp_best)
    os.remove(tmp_best)
    return model


if __name__ == "__main__":
    from pathlib import Path

    folder_path = "TicTacToe/Grok_Zero_Train/1"

    from TicTacToe.Build_Model import build_model
    from TicTacToe.game import TicTacToe
    from TicTacToe.config import build_config, train_config

    # game = Gomoku()
    game = TicTacToe()
    model = build_model(game.get_input_state().shape, game.policy_shape, build_config)
    # train(model, 5e-4, train_config,  str(Path(folder_path).parent), False)
