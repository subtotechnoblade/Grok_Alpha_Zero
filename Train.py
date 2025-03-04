import tensorflow as tf

from Net.Stablemax import Stablemax
from Net.Custom_Loss import Policy_Loss, Value_Loss, KLD
def train(train_dataloader, test_dataloader, model, learning_rate, build_config, train_config):
    # assume that save_folder path is Grok_Zero_Train/current_generation + 1
    kwargs = {"learning_rate": learning_rate,
              "beta_1": train_config["beta_1"],
              "beta_2": train_config["beta_2"],
              # "weight_decay": 1e-4,
              "gradient_accumulation_steps": train_config["gradient_accumulation_steps"],
              }

    if train_config["optimizer"].lower() == "adam":
        optimizer = tf.keras.optimizers.Adam(**kwargs)
    elif train_config["optimizer"].lower() == "adamw":
        optimizer = tf.keras.optimizers.AdamW(**kwargs)
    elif train_config["optimizer"].lower() == "nadam":
        optimizer = tf.keras.optimizers.Nadam(**kwargs)

    activation_fn = None
    if not train_config["use_gumbel"]:
        if build_config["use_stable_max"]:
            activation_fn = Stablemax(name="policy")
        else:
            activation_fn = tf.keras.layers.Activation("softmax")

    model.compile(optimizer=optimizer,
                  loss={"policy": Policy_Loss(activation_fn=activation_fn), "value": Value_Loss()},
                  metrics={"policy": KLD()}
                  )
    model.fit(train_dataloader,
              validation_data=test_dataloader,
              epochs=train_config["train_epochs"])
    return model

if __name__ == "__main__":
    from pathlib import Path
    folder_path = "TicTacToe/Grok_Zero_Train/1"

    from TicTacToe.Build_Model import build_model
    from TicTacToe.Tictactoe import TicTacToe, build_config, train_config

    # game = Gomoku()
    game = TicTacToe()
    model = build_model(game.get_input_state().shape, game.policy_shape, build_config)
    train(model, 5e-4, train_config,  str(Path(folder_path).parent), False)





