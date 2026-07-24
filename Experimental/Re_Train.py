import os
import tensorflow as tf
from Dataloader import Create_Dataset
from Train import train
def Train_NN(game_class, build_model_fn, build_config, train_config, generation, folder_path, save_folder_path):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


    gpu_devices = tf.config.list_physical_devices("GPU")
    for device in gpu_devices:
        tf.config.experimental.set_virtual_device_configuration(device, [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5700)])

    mixed_precision_policy = train_config.get("mixed_precision")
    if mixed_precision_policy is not None:
        policy = tf.keras.mixed_precision.Policy(mixed_precision_policy)
        tf.keras.mixed_precision.set_global_policy(policy)

    game = game_class()
    model = build_model_fn(game.get_input_state().shape, game.policy_shape, build_config, train_config)
    model.load_weights(f"{folder_path}/model.weights.h5")

    lr_decay = train_config["lr_decay"] ** (generation // train_config["decay_lr_after"])
    learning_rate = train_config["learning_rate"] * lr_decay

    print(f"Started training for generation: {generation} using lr = {learning_rate}!")
    train_dataloader, test_dataloader = Create_Dataset(str(Path(folder_path).parent),
                                                       num_previous_generations=train_config[
                                                           "num_previous_generations"],
                                                       train_batch_size=train_config["train_batch_size"],
                                                       test_batch_size=train_config["test_batch_size"],
                                                       train_percent=train_config["train_percent"],
                                                       train_decay=train_config["train_decay"],
                                                       test_percent=train_config["test_percent"],
                                                       test_decay=train_config["test_decay"], )
    model = train(train_dataloader, test_dataloader, model, learning_rate, build_config, train_config)
    Make_Generation_Folder(generation + 1)
    model.save_weights(f"{save_folder_path}/model.weights.h5") # overwrites the model