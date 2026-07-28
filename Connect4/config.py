"""Training hyperparameters for Connect4.

Kept in a separate file from game.py on purpose. numba stamps its on-disk cache
(@njit(cache=True)) on the source file's mtime and size, so any edit to the file
holding the jitted kernels throws the whole cache away - measured at ~5s of
recompile per process, multiplied by every self play worker. Tuning a value here
leaves game.py untouched, so the kernels stay cached.

Nothing heavy may be imported here: the self play workers load this module.
"""

build_config = {"num_resnet_layers": 4,  # This is the total amount of resnet layers in the model that are used
                "num_filters": 64,
                "use_stablemax": False,  # use stablemax, which will also use stablemax crossentropy

                "mixed_precision": None,  # None for no mixed precision, mixed_float16 for float16
          }
train_config = {
    "total_generations": 100,  # Total amount of generations, the training can be stopped and resume at any moment
    # a generation is defined by a round of self play, padding the dataset, model training, converting to onnx

    # Self Play variables
    "games_per_generation": 500,  # amount of self play games until we re train the network
    "max_actions": 42,  # Note that this should be less than max actions,
    "num_explore_actions_first": 3,  # A good rule of thumb is how long the opening should be for player -1
    "num_explore_actions_second": 3,  # Since player 1 is always at a disadvantage, we explore less and attempt to play better moves
    # ^ these two together set how many PLIES are sampled at tau=1: first + second.
    # alpha zero used ~15% of the average game length (30 plies out of ~200). Self play
    # averages ~32 plies here, so 6 plies is ~19%. At the old 8/7 it was 15 plies = 47%,
    # which meant most games opened off the centre column and were drawn by construction

    "use_gpu": True,  # Change this to False to use CPU for self play and inference
    "use_tensorrt": True,  # Assuming use_gpu is True, uses TensorrtExecutionProvider
    # change this to False to use CUDAExecutionProvider
    "trt_fp16": True,  # let tensorrt pick fp16 kernels at engine build time. The onnx model
    # stays fp32, this is only a permission for TRT to run layers in half precision where
    # that is faster. Independent of build_config["mixed_precision"], which is TRAINING only.
    # Measured on the rtx 2060: batch-1 inference 1244us (CUDA fp32) -> 698us (TRT fp32) ->
    # 624us (TRT fp16), and 99.8% top-1 policy agreement vs fp32.
    "use_inference_server": True,  # if an extremely large model is used, because of memory constraints, set this to True
    "max_cache_depth": 10,  # maximum depth in the search of the neural networks outputs we should cache, use this if the inference speed is under 1000it/s
    # ^ this is now an in-memory dict per worker (Session_Cache.py), not a disk cache, so
    # depth can be generous. Only plies < this are memoised, since only the opening region
    # repeats across the games a worker plays. At the old value of 1 it cached roughly 8
    # positions per generation, which is why the disk cache was very nearly a no-op.
    # Each worker prints its hit rate at the end of the generation, tune from that.
    "num_workers": 10,  # Number of multiprocessing workers used to self play

    # MCTS variables
    "MCTS_iteration_limit": 250,  # The number of iterations MCTS runs for. Should be 2 to 10x the number of starting legal moves
    # True defaults to iteration_limit = 3 * len(starting legal actions)
    "MCTS_time_limit": None,  # Not recommended to use for training, True defaults to 30 seconds
    "use_njit": None,  # None will automatically infer what is supposed to be use for windows/linux

    "use_gumbel": False,  # use gumbel according to https://openreview.net/pdf?id=bERaNdoegnO
    # These params will only be used when use_gumbel is set to True
    "m": 7,  # Number of actions sampled in the first stage of sequential halving
    "c_visit": 50.0,
    "c_scale": 1.0,

    # "opening_actions": [[0, 0.1], [1, 0.1], [2, 0.1], [3, 0.1], [4, 0.1], [5, 0.1], [6, 0.1]],

    # These params will be used when use_gumbel is set to False
    "c_puct_init": 1.5,  # (shouldn't change) Exploration constant lower -> exploitation, higher -> exploration
    # ^ this controls WIDTH, not "go check that one line". Search depth is roughly
    # ln(iterations) / ln(effective branching factor), so spreading visits costs depth.
    # measured from the gen 9 policy targets the search was hedging over 2.53 moves,
    # which is only ~7 plies of sight at 675 iterations. root dirichlet noise is the
    # mechanism for reconsidering a move the prior dismissed, not c_puct
    "dirichlet_alpha": 1.33,  # should be around (10 / average moves per given position)

    "q_weight": 0.20,  # value target = (1 - q_weight) * z + q_weight * q
    # 0.0 is the original alpha zero target (final outcome only, noisy),
    # 1.0 is the search's own root Q (suffers from the horizon effect), 0.5 averages them

    # "opening_actions": [[3, 0.5]], # starting first move in the format [[action1, prob0], [action1, prob1], ...],
    # if prob doesn't add up to 1, then the remaining prob is for the MCTS move

    "num_previous_generations": 4,  # The previous generation's data that will be used in training
    "train_percent": 1.0,  # The percent used fr training after the test set is taken
    "train_decay": 0.1,  # The decay rate for previous generations of data previous_train_percent = current_train_percent * train_decay
    "test_percent": 0.1,  # The percent of a dataset that will be used for validation
    "test_decay": 0.01,  # The decay rate for previous generations of data previous_test_percent = current_test_percent * test_decay
    "dedup_average": True,  # merge byte-identical states at load time, targets become the occurrence-weighted
    # mean. Kills the duplicate-label noise (~66% of samples are copies, see tmp/dup_audit.py) and the
    # opening-vs-midgame loss-weight imbalance. Applied per split side, pooled across the generation window.

    "train_batch_size": 1024,  # The number of samples in a batch for training in parallel
    "test_batch_size": None,  # If none, then train_batch_size will be used for the test batch size
}

def lr_schedule(generation):
    lr_decay = 0.2 ** (generation // 25)  # 3e-3 -> 6e-4 -> 1.2e-4 -> 2.4e-5 over 100 generations
    return 3e-3 * lr_decay
    # base was 1e-2. See tmp/exp_c_seeds_lr.py and tmp/exp_d_lr_curves.py: over 3 seeds the
    # best val loss basin is flat from 1e-3 to 3e-3 and degrades hard by 1e-2 (2.63 -> 2.89).
    # That held BEFORE the muon reshape fix too, so 1e-2 was always too hot. The fix makes
    # trunk steps ~6x larger at the same nominal lr, which is a second reason to come down.
    # Caveat: measured on a 6k position / 6 epoch proxy, not a full generation.


optimizer_config = {
    "optimizer": "muon", # ["adam", "nadam", "muon"]
    # will be passed directly into the optimizer
    "gradient_accumulation_steps": None,
    "train_epochs": 8,  # The number of epochs for training
    # ^ connect4 augments 2x, so 8 epochs is 16 gradient exposures per unique position per
    # generation. Now that the val split is honest, ModelCheckpoint(save_best_only) really
    # does pick the best epoch, so this is a compute bound rather than an overfitting bound
    "kwargs": {
        "learning_rate": lr_schedule,
        "weight_decay": 1e-4,
        "use_nadam": False,
        "adam_beta_1": 0.9,  # beta_1 for adam/nadam for muon
        "adam_beta_2": 0.995,  # beta_2 for adam/nadam for muon
        "muon_beta": 0.95,
        "exclude_layers": ["eyes", "policy_1",  "value_1"]
    },
    "use_orthograd": False,
    # OrthoGrad removes the component of the gradient parallel to W, to stop the weights
    # inflating. It cannot do that here. Every conv feeds a BatchNorm, which makes the loss
    # scale invariant in that kernel, so the gradient is ALREADY perpendicular to W
    # (measured cos = 0.00000) and there is nothing to project out. The norm still grows
    # because perpendicular steps add in quadrature, sqrt(|W0|^2 + N*d^2), which no
    # projection can prevent. Measured on CIFAR-10, 3 seeds, paired: Orthograd(Muon) was
    # worse on every seed. See tmp/REPORT.md section 1.
}
