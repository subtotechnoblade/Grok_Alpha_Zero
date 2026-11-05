"""Training hyperparameters for Gomoku."""
from Config import AlphaZeroConfig


build_config = {"num_resnet_layers": 4,
                "num_filters": 128,

                "use_stablemax": False,
                "mixed_precision": "mixed_float16",  # None, or mixed_float16
                "bn_momentum": 0.9,
                "global_pool_blocks": (1, 3),
                "pool_filters": 32,
          }


class GomokuTrain(AlphaZeroConfig):
    # ---- the training loop ----
    total_generations = 100
    initial_games_per_generation = 2000
    games_per_generation = 1000
    initial_positions_per_generation = 70000
    positions_per_generation = 30000
    max_actions = 225

    # ---- self play workers / inference ----
    num_workers = 24
    use_gpu = True
    use_tensorrt = True
    trt_fp16 = True
    use_inference_server = True
    nn_input_dtype = "uint8"
    nn_output_dtype = "float16"
    max_cache_depth = 3
    symmetric_cache_evals = True
    random_symmetry_evals = True
    dml_device_id = None  # 0 = 760M igpu, 1 = 2060

    # ---- search ----
    MCTS_iteration_limit = 2000
    MCTS_time_limit = None
    leaves_per_batch = 16
    virtual_loss = 1.0
    use_njit = None
    create_new_root = False

    # ---- exploration ----
    num_explore_actions_first = 4
    num_explore_actions_second = 3
    c_puct_init = 2.0
    fpu_reduction = 0.2
    dirichlet_alpha = 0.05  # ~10 / 225 legal moves
    dirichlet_epsilon = 0.25

    # ---- training targets ----
    q_weight = 0.25
    aux_heads = True
    soft_policy = True
    opponent_policy = True
    soft_policy_temperature = 4.0
    wdl_head = True
    value_discount = 0.99
    full_search_prob = 0.25
    cheap_sims = 320
    escalate_policy = True
    escalate_value = True
    escalate_disagreement = 0.693
    escalate_value_gap = 0.3
    cheap_policy_targets = True
    cheap_target_weight = 0.25
    surprise_weighting = True
    opening_positions = [[[[7, 7], [8, 7], [8, 3]], 0.5]]  # H8 I8 I12
    # opening_actions = [[[7, 7], 0.25],
    #                    [[6, 6], 0.03], [[7, 6], 0.03], [[8, 6], 0.03], [[6, 7], 0.03],
    #                    [[8, 7], 0.03], [[6, 8], 0.03], [[7, 8], 0.03], [[8, 8], 0.03]]

    # ---- dataset window ----
    num_previous_generations = 5
    train_percent = 1.0
    train_decay = 0.5
    test_percent = 0.1
    test_decay = 0.025
    dedup_average = True
    max_sample_weight = 32
    train_batch_size = 1024
    test_batch_size = None


train_config = GomokuTrain()


PEAK_LR = 2.5e-3
LR_DROP = 1.5
LR_DROP_EVERY = 2
LR_FLOOR = 3.75e-4


def lr_schedule(generation):
    from Net.Schedules import step_decay_lr
    return step_decay_lr(generation, PEAK_LR, drop=LR_DROP, every=LR_DROP_EVERY, floor_lr=LR_FLOOR)


optimizer_config = {
    "optimizer": "nadam",  # ["adam", "nadam", "muon", "sgd"]
    "gradient_accumulation_steps": None,
    "train_epochs": 3,
    "value_loss_weight": 0.25,
    "soft_policy_loss_weight": 1.0,
    "opponent_policy_loss_weight": 0.15,
    "aux_loss_weight": 0.25,
    "ema_decay": 0.99,
    "ema_bn_batches": 50,
    "checkpoint_value_weight": 0.1,  # value's weight in val_norm only; omit to use value_loss_weight
    "kwargs": {
        "learning_rate": lr_schedule,
        "weight_decay": 4e-3,
    },
    "use_orthograd": False,
    "orthograd_exclude": ["eyes", "policy_2", "value_3"],

    "lr_warmup_frac": 0.05,
    "lr_floor_frac": 0.1,  # 1.0 = warmup then hold at peak, no decay
}
