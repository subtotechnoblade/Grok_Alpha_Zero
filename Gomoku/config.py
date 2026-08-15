"""Training hyperparameters for Gomoku."""
from Config import AlphaZeroConfig


build_config = {"num_resnet_layers": 4,
                "num_filters": 128,
                "rr_alpha": 0.05,

                "use_stablemax": False,
                "mixed_precision": None,  # None, or mixed_float16
          }


class GomokuTrain(AlphaZeroConfig):
    # ---- the training loop ----
    total_generations = 100
    initial_games_per_generation = 10000
    games_per_generation = 5000
    max_actions = 150

    # ---- self play workers / inference ----
    num_workers = 8
    use_gpu = True
    use_tensorrt = True
    trt_fp16 = True
    use_inference_server = True
    max_cache_depth = 3
    dml_device_id = None  # 0 = 760M igpu, 1 = 2060

    # ---- search ----
    MCTS_iteration_limit = 1000
    MCTS_time_limit = None
    use_njit = None
    create_new_root = True

    # ---- exploration ----
    num_explore_actions_first = 3
    num_explore_actions_second = 3
    c_puct_init = 4.5
    dirichlet_alpha = 0.05  # ~10 / 225 legal moves
    dirichlet_epsilon = 0.25

    # ---- training targets ----
    q_weight = 0.5
    opening_actions = [[[7, 7], 0.333],
                       [[6, 6], 0.05], [[7, 6], 0.05], [[8, 6], 0.05], [[6, 7], 0.05],
                       [[8, 7], 0.05], [[6, 8], 0.05], [[7, 8], 0.05], [[8, 8], 0.05]]

    # ---- dataset window ----
    num_previous_generations = 3
    train_percent = 1.0
    train_decay = 0.2
    test_percent = 0.1
    test_decay = 0.03
    dedup_average = True
    train_batch_size = 1024
    test_batch_size = 256


train_config = GomokuTrain()


PEAK_LR = 5e-3
LR_DROP = 5.0
LR_DROP_EVERY = 20


def lr_schedule(generation):
    from Net.Schedules import step_decay_lr
    return step_decay_lr(generation, PEAK_LR, drop=LR_DROP, every=LR_DROP_EVERY)


optimizer_config = {
    "optimizer": "muon",  # ["adam", "nadam", "muon"]
    "gradient_accumulation_steps": None,
    "train_epochs": 15,
    "kwargs": {
        "learning_rate": lr_schedule,
        "weight_decay": 0.0,
        "adam_beta_1": 0.9,
        "adam_beta_2": 0.995,
        "muon_beta": 0.95,
        "exclude_layers": ["policy_2", "value_3"]
    },
    "use_orthograd": True,
    "orthograd_exclude": ["eyes", "policy_2", "value_3"],

    "lr_warmup_frac": 0.05,
    "lr_floor_frac": 1.0,  # 1.0 = warmup then hold at peak, no decay
}
