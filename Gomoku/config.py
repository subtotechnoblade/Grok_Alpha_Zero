"""Training hyperparameters for Gomoku.

Kept in a separate file from game.py on purpose. numba stamps its on-disk cache
(@njit(cache=True)) on the source file's mtime and size, so any edit to the file
holding the jitted kernels throws the whole cache away - measured at ~5s of
recompile per process, multiplied by every self play worker. Tuning a value here
leaves game.py untouched, so the kernels stay cached.

Nothing heavy may be imported here: the self play workers load this module.
"""

# This is for building the model
build_config = {"num_resnet_layers": 4,  # This is the total amount of resnet layers in the model that are used
                "num_filters": 128,
                "rr_alpha": 0.05,

                "use_stablemax": False,
                "mixed_precision": None,  # None for no mixed precision, else mixed_float16 for float16
          }

train_config = {
    "total_generations": 100,  # Total amount of generations, the training can be stopped and resume at any moment
    # a generation is defined by a round of self play, padding the dataset, model training, converting to onnx

    # Self Play variables
    "games_per_generation": 2000,  # amount of self play games until we re train the network
    "max_actions": 150,  # Note that this should be less than max actions,
    "num_explore_actions_first": 3,  # A good rule of thumb is how long the opening should be for player -1
    "num_explore_actions_second": 3,  # Since player 1 is always at a disadvantage, we explore less and attempt to play better moves
    # ^ these two together set how many PLIES are sampled at tau=1: first + second.
    # alpha zero used ~15% of the average game length (30 plies out of ~200). Self play
    # currently averages ~21 plies, so 6 plies is ~29% and should come down further as
    # games lengthen. At the old 6/4 it was 10 plies = ~50%, and in gomoku one loose
    # opening move loses outright, so z was mostly recording who blundered while exploring

    "use_gpu": True,  # Change this to False to use CPU for self play and inference
    "use_tensorrt": True,  # Assuming use_gpu is True, uses TensorrtExecutionProvider
    # change this to False to use CUDAExecutionProvider
    "use_inference_server": True,  # if an extremely large model is used, because of memory constraints, set this to True
    "max_cache_depth": 3,  # maximum depth in the search of the neural networks outputs we should cache
    "num_workers": 8,  # Number of multiprocessing workers used to self play

    # MCTS variables
    "MCTS_iteration_limit": 1000,  # The number of iterations MCTS runs for. Should be 2 to 10x the number of starting legal moves
    # True defaults to iteration_limit = 3 * len(starting legal actions)
    "MCTS_time_limit": None,  # Not recommended to use for training, True defaults to 30 seconds
    "use_njit": None,  # None will automatically infer what is supposed to be use for windows/linux

    "use_gumbel": False,  # use gumbel according to https://openreview.net/pdf?id=bERaNdoegnO
    # These params will only be used when use_gumbel is set to True
    "m": 64,  # Number of actions sampled in the first stage of sequential halving
    "c_visit": 50.0,
    "c_scale": 1.0,

    # These params will be used when use_gumbel is set to False
    "c_puct_init": 4.5,  # (shouldn't change) Exploration constant lower -> exploitation, higher -> exploration
    "dirichlet_alpha": 0.05,  # should be around (10 / average moves per position)

    "q_weight": 0.5,  # value target = (1 - q_weight) * z + q_weight * q
    # 0.0 is the original alpha zero target (final outcome only, noisy),
    # 1.0 is the search's own root Q (suffers from the horizon effect), 0.5 averages them

    "opening_actions": [[[7, 7], 0.333],
                        [[6, 6], 0.05], [[7, 6], 0.05], [[8, 6], 0.05], [[6, 7], 0.05], [[8, 7], 0.05], [[6, 8], 0.05], [[7, 8], 0.05], [[8, 8], 0.05]
                        ], # starting first move in the format [[action1, prob0], [action1, prob1], ...],
    # if prob doesn't add up to 1, then the remaining prob is for the MCTS move

    "num_previous_generations": 3,  # The previous generation's data that will be used in training
    "train_percent": 1.0,  # The percent used for training after the test set is taken
    "train_decay": 0.2,  # The decay rate for previous generations of data previous_train_percent = current_train_percent * train_decay
    "test_percent": 0.1,  # The percent of a dataset that will be used for validation
    "test_decay": 0.03,  # The decay rate for previous generations of data previous_test_percent = current_test_percent * test_decay

    "train_batch_size": 1024,  # The number of samples in a batch for training in parallel
    "test_batch_size": 256,  # If none, then train_batch_size will be used for the test batch size
}
def lr_schedule(generation):
    lr_decay = 0.2 ** (generation // 20) # 15 is decay after
    return 5e-3 * lr_decay


optimizer_config = {
    "optimizer": "muon", # ["adam", "nadam", "muon"]
    # will be passed directly into the optimizer
    "gradient_accumulation_steps": None,
    "train_epochs": 15,  # The number of epochs for training
    "kwargs":{
        "learning_rate": lr_schedule,
        "weight_decay": 0.0,
        "adam_beta_1": 0.9,  # beta_1 for adam/nadam for muon
        "adam_beta_2": 0.995,  # beta_2 for adam/nadam for muon
        "muon_beta": 0.95,
        "exclude_layers": ["policy_2", "value_3"]
    },
    "use_orthograd": True,
}
