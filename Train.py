import tensorflow as tf

from Net.Stablemax import Stablemax
from Net.Alpha_Loss import Policy_Loss, Value_Loss, KLD
from Net.Gumbel_Loss import Policy_Loss_Gumbel, Value_Loss_Gumbel, KLD_Gumbel
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

    if not train_config["use_gumbel"]:
        policy_loss, value_loss, kld = Policy_Loss(), Value_Loss(), KLD()
    else:
        if build_config["use_stablemax"]:
            activation_fn = Stablemax(dtype="float32")
        else:
            activation_fn = tf.keras.layers.Activation("softmax", dtype="float32")
        policy_loss, value_loss, kld = (Policy_Loss_Gumbel(activation_fn=activation_fn, dtype="float32"),
                                        Value_Loss_Gumbel(dtype="float32"),
                                        KLD_Gumbel(activation_fn=activation_fn, dtype="float32"))


    model.compile(optimizer=optimizer,
                  loss={"policy": policy_loss, "value": value_loss},
                  metrics={"policy": kld},
                  auto_scale_loss=True if train_config.get("mixed_precision") == "mixed_float16" else False
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





