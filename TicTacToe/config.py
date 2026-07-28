"""Training hyperparameters for TicTacToe.

Kept in a separate file from game.py on purpose. numba stamps its on-disk cache
(@njit(cache=True)) on the source file's mtime and size, so any edit to the file
holding the jitted kernels throws the whole cache away - measured at ~5s of
recompile per process, multiplied by every self play worker. Tuning a value here
leaves game.py untouched, so the kernels stay cached.

Nothing heavy may be imported here: the self play workers load this module.
"""

build_config = {"num_resnet_layers": 2,

                "use_stablemax": False,
                "use_grok_fast": True,
                "use_orthograd": True,
                "grok_fast_lambda": 4.5,  # This is for grok fast, won't be used if model is Grok_Fast_EMA_Model
                }

train_config = {
    "total_generations": 15,  # Total number of generations, the training can be stopped and resume at any moment
    # a generation is defined by a round of self play, padding the dataset, model training, converting to onnx

    # Self Play variables
    "games_per_generation": 100,  # number of self play games until we re train the network
    "max_actions": 9,  # Note that this should be
    "num_explore_actions_first": 2,
    # This is for tictactoe, a good rule of thumb is 10% to 20% of the average length of a game
    "num_explore_actions_second": 2,
    # for a random player player -1 almost always wins, so player 1 should try playing the best move

    "use_gpu": True,  # Change this to false to use CPU for self play and inference
    "use_tensorrt": False,  # Assuming use_gpu is True, uses TensorrtExecutionProvider
    # change this to False to use CUDAExecutionProvider
    "use_inference_server": True,
    # if an extremely large model is used, because of memory constraints, set this to True
    "max_cache_depth": 0,  # maximum depth in the search of the neural networks outputs we should cache
    "num_workers": 8,  # Number of multiprocessing workers used to self play

    # MCTS variables
    "MCTS_iteration_limit": 30,  # The number of iterations MCTS runs for. Should be 2 to 10x the number of starting legal moves
    # True defaults to iteration_limit = 3 * len(starting legal actions)
    "MCTS_time_limit": None,  # Not recommended to use for training, True defaults to 30 seconds
    "use_njit": None,  # None will automatically infer what is supposed to be use for windows/linux

    "use_gumbel": False,  # use gumbel according to https://openreview.net/pdf?id=bERaNdoegnO, time_limit won't be used
    # These params will only be used when use_gumbel is set to True
    "m": 2,  # Number of actions sampled in the first stage of sequential halving
    "c_visit": 50.0,
    "c_scale": 2.0,

    # These params will be used when use_gumbel is set to False
    "c_puct_init": 2.5,  # (shouldn't change) Exploration constant lower -> exploitation, higher -> exploration
    "dirichlet_alpha": 1.0,  # should be around (10 / average moves per game)

    # "opening_actions": [[[1, 1], 0.4]],  # starting first move in the format [[action1, prob0], [action1, prob1], ...],
    # # if prob doesn't add up to 1, then the remaining prob is for the MCTS move

    "num_previous_generations": 3,  # The previous generation's data that will be used in training
    "train_percent": 1.0,  # The percent used for training after the test set is taken
    "train_decay": 0.5,
    # The decay rate for previous generations of data previous_train_percent = current_train_percent * train_decay
    "test_percent": 0.1,  # The percent of a dataset that will be used for validation
    "test_decay": 0.5,
    # The decay rate for previous generations of data previous_test_percent = current_test_percent * test_decay

    "mixed_precision": None,  # None for no mixed precision, mixed_float16 for float16
    "train_batch_size": 512,  # The number of samples in a batch for training in parallel
    "test_batch_size": None,  # If none, then train_batch_size will be used for the test batch size
    "gradient_accumulation_steps": None,
    "learning_rate": 1e-3,  # Depending on how many layers you use. Recommended to be between 1e-3 to 5e-4
    "decay_lr_after": 3,  # When the n generations pass,... learning rate will be decreased by lr decay
    "lr_decay": 0.75,  # multiplies this to learning rate every decay_lr_after
    "beta_1": 0.9,  # DO NOT TOUCH unless you know what you are doing
    "beta_2": 0.995,  # DO NOT TOUCH. This determines whether it groks or not. Hovers between 0.98 to 0.995
    "optimizer": "Nadam",  # optimizer options are ["Adam", "AdamW", "Nadam"]
    "train_epochs": 15,  # The number of epochs for training
}
