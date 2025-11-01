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
def train(train_dataloader, test_dataloader, model, learning_rate, configs: list[dict]):
    build_config, train_config, optimizer_config = configs
    # assume that save_folder path is Grok_Zero_Train/current_generation + 1
    accum_steps = optimizer_config.get("gradient_accumulation_steps")
    if accum_steps is not None and accum_steps >= 2:
        learning_rate /= accum_steps # this is to counter the gradients being scaled by accum_steps thus we divide

    optimizer = {"adam": Adam, "nadam": Nadam, "muon": Muon}[optimizer_config["optimizer"].lower()]
    optimizer_config["kwargs"]["learning_rate"] = learning_rate # this is to overwrite learning rate in kwargs if it was a function
    optimizer = optimizer(**optimizer_config["kwargs"])

    # order matters, grads -> orthograd -> grokfast EMA filter -> optimizer
    # orthograd(grokfast_EMA(base_optimizer))
    if optimizer_config["use_grokfast"]:
        optimizer = Grokfast_EMA(optimizer, lamb=optimizer_config["grokfast_lambda"])

    if optimizer_config["use_orthograd"]:
        optimizer = Orthograd(optimizer)

    if not train_config["use_gumbel"]:
        policy_loss, value_loss, kld = Policy_Loss(dtype="float32"), Value_Loss(dtype="float32"), KLD(dtype="float32")
    else:
        if build_config["use_stablemax"]:
            activation_fn = Stablemax(dtype="float32")
        else:
            activation_fn = tf.keras.layers.Activation("softmax", dtype="float64")
            # activation_fn = Taylor_Softmax(name="softmax", dtype="float64")
        policy_loss, value_loss, kld = (Policy_Loss_Gumbel(activation_fn=activation_fn, dtype="float64"),
                                        Value_Loss_Gumbel(dtype="float64"),
                                        KLD_Gumbel(activation_fn=activation_fn, dtype="float64"))


    model.compile(optimizer=optimizer,
                  loss={"policy": policy_loss, "value": value_loss},
                  metrics={"policy": kld})

    save_best_model = tf.keras.callbacks.ModelCheckpoint(
        filepath='Grok_Zero_Train/tmp_best.weights.h5',  # Save path (e.g., .keras, .h5, or directory)
        monitor='val_loss',  # Monitor validation loss
        save_best_only=True,  # Save ONLY the best model
        save_weights_only=True,
        mode='min',  # 'min' for val_loss (lower = better)
        verbose=1  # Optional: show message when saving
    )

    model.fit(train_dataloader,
              validation_data=test_dataloader,
              epochs=optimizer_config["train_epochs"],
              callbacks=[save_best_model])

    model.load_weights("Grok_Zero_Train/tmp_best.weights.h5")
    os.remove("Grok_Zero_Train/tmp_best.weights.h5")
    return model

if __name__ == "__main__":
    from pathlib import Path
    folder_path = "TicTacToe/Grok_Zero_Train/1"

    from TicTacToe.Build_Model import build_model
    from TicTacToe.Tictactoe import TicTacToe, build_config, train_config

    # game = Gomoku()
    game = TicTacToe()
    model = build_model(game.get_input_state().shape, game.policy_shape, build_config)
    # train(model, 5e-4, train_config,  str(Path(folder_path).parent), False)





