import tensorflow as tf

from Net.Custom_Loss import Policy_Loss, Value_Loss, KLD
from Net_Time_Parallel.Custom_Loss_Time_Parallel import Policy_Loss_Time_Parallel, Value_Loss_Time_Parallel, KLD_Time_Parallel
def train(train_dataloader, test_dataloader, model, learning_rate, train_config, parent_path):
    # assume that save_folder path is Grok_Zero_Train/current_generation + 1
    # train_dataset, test_dataset = Create_Dataset(parent_path,
    #                                              num_previous_generations=train_config["num_previous_generations"],
    #                                              train_batch_size=train_config["train_batch_size"],
    #                                              test_batch_size=train_config["test_batch_size"],
    #                                              train_percent=train_config["train_percent"],
    #                                              train_decay=train_config["train_decay"],
    #                                              test_percent=train_config["test_percent"],
    #                                              test_decay=train_config["test_decay"],)

    kwargs = {"learning_rate": learning_rate,
              "beta_1": train_config["beta_1"],
              "beta_2": train_config["beta_2"],
              # "weight_decay": 1e-4,
              "gradient_accumulation_steps": train_config["gradient_accumulation_steps"],
              }

    if train_config["use_time_parallel"]:
        policy_loss = Policy_Loss_Time_Parallel()
        value_loss = Value_Loss_Time_Parallel()
        kld = KLD_Time_Parallel()
    else:
        policy_loss = Policy_Loss()
        value_loss = Value_Loss()
        kld = KLD()

    if train_config["optimizer"].lower() == "adam":
        optimizer = tf.keras.optimizers.Adam(**kwargs)
    elif train_config["optimizer"].lower() == "adamw":
        optimizer = tf.keras.optimizers.AdamW(**kwargs)
    elif train_config["optimizer"].lower() == "nadam":
        optimizer = tf.keras.optimizers.Nadam(**kwargs)

    model.compile(optimizer=optimizer,
                  loss={"policy": policy_loss, "value": value_loss},
                  metrics={"policy": kld}
                  )
    model.fit(train_dataloader,
              validation_data=test_dataloader,
              epochs=train_config["train_epochs"])
    return model

if __name__ == "__main__":
    from pathlib import Path
    folder_path = "TicTacToe/Grok_Zero_Train/1"
    from Gomoku.Build_Model_Time_Parallel import build_model

    from TicTacToe.Tictactoe import TicTacToe, build_config, train_config
    from TicTacToe.Build_Model_Time_Parallel import build_model

    # game = Gomoku()
    game = TicTacToe()
    model = build_model(game.get_input_state().shape, game.policy_shape, build_config)
    train(model, 5e-4, train_config,  str(Path(folder_path).parent), False)





