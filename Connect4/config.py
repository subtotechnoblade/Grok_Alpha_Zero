"""Training hyperparameters for Connect4."""
from Config import AlphaZeroConfig


build_config = {"num_resnet_layers": 4,
                "num_filters": 64,
                "use_stablemax": False,

                "mixed_precision": None,  # None, or mixed_float16
          }


class Connect4Train(AlphaZeroConfig):
    # ---- the training loop ----
    total_generations = 100
    games_per_generation = 500
    max_actions = 42

    # ---- self play workers / inference ----
    num_workers = 10
    use_gpu = True
    use_tensorrt = False
    trt_fp16 = True
    use_inference_server = True
    max_cache_depth = 10
    dml_device_id = 0  # 0 = 760M igpu, 1 = 2060

    # ---- search ----
    MCTS_iteration_limit = 250
    MCTS_time_limit = None
    use_njit = None
    create_new_root = False

    # ---- exploration ----
    num_explore_actions_first = 3
    num_explore_actions_second = 3
    c_puct_init = 1.5
    dirichlet_alpha = 1.33
    dirichlet_epsilon = 0.25

    # ---- training targets ----
    q_weight = 0.15
    # opening_actions = [[0, 0.05], [1, 0.05], [2, 0.05], [4, 0.05], [5, 0.05], [6, 0.05]]

    # ---- dataset window ----
    num_previous_generations = 5
    train_percent = 1.0
    train_decay = 0.1
    test_percent = 0.1
    test_decay = 0.01
    dedup_average = True
    train_batch_size = 1024
    test_batch_size = None


train_config = Connect4Train()


PEAK_LR = 1e-2
LR_DROP = 2.0
LR_DROP_EVERY = 5


def lr_schedule(generation):
    from Net.Schedules import step_decay_lr
    return step_decay_lr(generation, PEAK_LR, drop=LR_DROP, every=LR_DROP_EVERY)


optimizer_config = {
    "optimizer": "adam",  # ["adam", "nadam", "muon"]
    "gradient_accumulation_steps": None,
    "train_epochs": 8,
    "kwargs": {
        "learning_rate": lr_schedule,
        "weight_decay": 4e-3,
        # "use_nadam": False,
        # "adam_beta_1": 0.90,
        # "adam_beta_2": 0.995,
        # "muon_beta": 0.95,
        # "exclude_layers": ["eyes", "policy_1",  "value_1"]
    },
    "use_orthograd": False,
    "orthograd_exclude": ["eyes", "policy_1", "value_1"],

    "lr_warmup_frac": 0.05,
    "lr_floor_frac": 1.0,  # 1.0 = warmup then hold at peak, no decay
}
