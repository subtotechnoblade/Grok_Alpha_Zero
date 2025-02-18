import tensorflow as tf
from Gomoku.Gomoku import Gomoku, train_config

from Dataloader import Create_Dataset
from Net.Custom_Loss import Policy_Loss, Value_Loss
def train(model, learning_rate, folder_path, train_config, generation):
    # assume that folder path is Grok_Zero_Train/0
    # generation = is the current generation, next generation in where the new model should be saved is generation + 1
    train_dataset, test_dataset = Create_Dataset(folder_path,
                                                 num_previous_generations=train_config["num_previous_generation"],
                                                 train_batch_size=train_config["train_batch_size"],
                                                 test_batch_size=train_config["test_batch_size"],
                                                 train_percent=train_config["train_percent"],
                                                 train_decay=train_config["train_decay"],
                                                 test_percent=train_config["test_percent"],
                                                 test_decay=train_config["test_decay"],)

    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                                      beta_1=train_config["beta_1"],
                                                      beta_2=train_config["beta_2"],
                                                      weight_decay=1e-4),
                  loss={"policy": Policy_Loss(), "value": Value_Loss()},
                  #Brian will probably create custom accuracies for this
                  # todo make accuracies class
                  )
    model.fit(train_dataset,
              validation_data=test_dataset,
              epochs=train_config["train_epochs"])





