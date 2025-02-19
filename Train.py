import tensorflow as tf

from Dataloader import Create_Dataset
from Net.Custom_Loss import Policy_Loss, Value_Loss
def train(model, learning_rate, train_config, folder_path, save_folder_path):
    # assume that save_folder path is Grok_Zero_Train/current_generation + 1
    train_dataset, test_dataset = Create_Dataset(folder_path,
                                                 num_previous_generations=train_config["num_previous_generations"],
                                                 train_batch_size=train_config["train_batch_size"],
                                                 test_batch_size=train_config["test_batch_size"],
                                                 train_percent=train_config["train_percent"],
                                                 train_decay=train_config["train_decay"],
                                                 test_percent=train_config["test_percent"],
                                                 test_decay=train_config["test_decay"],)

    kwargs = {"learning_rate": learning_rate,
              "beta_1": train_config["beta_1"],
              "beta_2": train_config["beta_2"],
              "weight_decay": 1e-4
              }
    if train_config["optimizer"].lower() == "adam":
        optimizer = tf.keras.optimizers.Adam(**kwargs)
    elif train_config["optimizer"].lower() == "adamw":
        optimizer = tf.keras.optimizers.AdamW(**kwargs)
    elif train_config["optimizer"].lower() == "nadam":
        optimizer = tf.keras.optimizers.Nadam(**kwargs)

    model.compile(optimizer=optimizer,
                  loss={"policy": Policy_Loss(), "value": Value_Loss()},
                  #Brian will probably create custom accuracies for this
                  # todo make accuracies class
                  jit_compile=train_config["use_gpu"]
                  )
    model.fit(train_dataset,
              validation_data=test_dataset,
              epochs=train_config["train_epochs"])
    if save_folder_path:
        model.save_weights(f"{save_folder_path}/model.weights.h5")

if __name__ == "__main__":
    folder_path = "TicTacToe/Grok_Zero_Train/0"
    from Gomoku.Gomoku import Gomoku, build_config, train_config
    from Gomoku.Build_Model import build_model

    from TicTacToe.Tictactoe import TicTacToe, build_config, train_config
    from TicTacToe.Build_Model import build_model

    # game = Gomoku()
    game = TicTacToe()
    model = build_model(game.get_input_state().shape, game.policy_shape, build_config)
    train(model, 5e-4, train_config, folder_path, False)





