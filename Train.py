import os

import tensorflow as tf

from Net.Stablemax import Stablemax
from Net.Taylor_Softmax import Taylor_Softmax
from Net.Alpha_Loss import Policy_Loss, Value_Loss, KLD
from Net.Gumbel_Loss import Policy_Loss_Gumbel, Value_Loss_Gumbel, KLD_Gumbel
def train(train_dataloader, test_dataloader, model, learning_rate, build_config: dict, train_config:dict):
    # assume that save_folder path is Grok_Zero_Train/current_generation + 1
    accum_steps = train_config.get("gradient_accumulation_steps")
    if accum_steps is not None and accum_steps >= 2:
        learning_rate /= accum_steps # this is to counter the gradients being scaled by accum_steps  thus we divide
    kwargs = {"learning_rate": learning_rate,
              "beta_1": train_config["beta_1"],
              "beta_2": train_config["beta_2"],
              "weight_decay": 1e-2,
              "gradient_accumulation_steps": train_config["gradient_accumulation_steps"],
              "epsilon": 1e-15,
              }

    if train_config["optimizer"].lower() == "adam":
        optimizer = tf.keras.optimizers.Adam(**kwargs)
    elif train_config["optimizer"].lower() == "adamw":
        optimizer = tf.keras.optimizers.AdamW(**kwargs)
    elif train_config["optimizer"].lower() == "nadam":
        optimizer = tf.keras.optimizers.Nadam(**kwargs)

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
              epochs=train_config["train_epochs"],
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





