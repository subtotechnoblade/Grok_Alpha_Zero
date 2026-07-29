"""Training hyperparameters for Connect4.

Kept in a separate file from game.py on purpose. numba stamps its on-disk cache
(@njit(cache=True)) on the source file's mtime and size, so any edit to the file
holding the jitted kernels throws the whole cache away - measured at ~5s of
recompile per process, multiplied by every self play worker. Tuning a value here
leaves game.py untouched, so the kernels stay cached.

Nothing heavy may be imported here: the self play workers load this module.

train_config is a typed template (see Config.py): every assignment below is
checked at import - a typo'd name or invalid value errors immediately, and
gumbel parameters cannot be set on this class (switching algorithms means
subclassing GumbelAZConfig instead). train_config.explain() prints this table
with meanings (pipeline.py does it at startup); `python Config.py` regenerates
the markdown reference.
"""
from Config import AlphaZeroConfig


build_config = {"num_resnet_layers": 4,  # This is the total amount of resnet layers in the model that are used
                "num_filters": 64,
                "use_stablemax": False,  # use stablemax, which will also use stablemax crossentropy

                "mixed_precision": None,  # None for no mixed precision, mixed_float16 for float16
          }
# build_config stays a free-form dict on purpose: it is the contract between a
# developer and THEIR build_model, so there is nothing to standardize.


class Connect4Train(AlphaZeroConfig):
    # ---- the training loop ----
    total_generations = 100
    games_per_generation = 500  # connect4 is small, 1000 games/gen was pure cost
    max_actions = 42  # a 6x7 board fills at ply 42. A game can be WON on the 42nd
    # move - Self_Play only scores a draw if nobody won (the old unconditional
    # cutoff mislabeled 188 wins as draws, all stolen from player +1)

    # ---- self play workers / inference ----
    num_workers = 10  # measured on this 12-core machine: 8-12 is the flat optimum
    # since the mp.Event fix, the old cpu_count()//2 clamp is gone
    use_gpu = True
    use_tensorrt = False  # False while running on the 760M iGPU (AMD = DirectML
    # only, tensorflow_updated env). Self play speed is measured IDENTICAL to the
    # 2060 - the pipeline is latency-bound, not GPU-bound. Flip back to True on
    # the 2060 in the tf_trt env: TRT fp16 batch-1 573us and FLAT to batch 32
    # (526us) vs DirectML 366us growing to 753us - TRT wins above batch ~14 and
    # is the right bet once bigger server batches / virtual loss land
    trt_fp16 = True
    use_inference_server = True
    max_cache_depth = 10  # ply < 10 memo. At 15 the hit-rate gain was negligible
    # for connect4's short openings and the dict grows ~2x. Workers print their
    # hit rate each generation, tune from that
    dml_device_id = None  # only matters without tensorrt: 0 = 760M igpu, 1 = 2060

    # ---- search ----
    MCTS_iteration_limit = 250  # Self_Play runs int(1.5x) = 375 sims per move.
    # Depth is ~ln(sims)/ln(effective branching): measured 2.53 eff. branching
    # -> ~7 plies of sight. More sims deepen slowly; a better value net is depth
    MCTS_time_limit = None
    use_njit = None
    create_new_root = True  # fresh tree per self play move -> honest targets.
    # With reuse, 48-70% of the recorded policy target was inherited visits
    # (the gen 7-9 col-1 ratchet). Play-time reuse in play.py is unaffected

    # ---- exploration (nearly the ONLY entropy source in self play) ----
    num_explore_actions_first = 3   # tau=1 plies for player -1: plies 0,2,4.
    # ~19% of the 32-ply average game. At the old 8/7 it was 47% of the game and
    # most games opened off-centre and drew by construction
    num_explore_actions_second = 3  # plies 1,3,5 (off-by-one fixed - was 2 plies)
    c_puct_init = 1.5  # width, not depth: spreading visits costs sight. Root
    # dirichlet is the mechanism for reconsidering a dismissed move, not this
    dirichlet_alpha = 1.33  # ~10/7 legal moves
    dirichlet_epsilon = 0.25

    # ---- training targets ----
    q_weight = 0.2  # target = 0.8*z + 0.2*rootQ; z is noisy, q is horizon-blind
    opening_actions = []  # partial book, e.g. [[3, 0.5]] = 50% forced centre.
    # Keep it PARTIAL: the 100% uniform book produced the always-draw network
    # (E[z|empty] = 0 exactly). Candidate cure if col-3 doesn't recover post-fixes

    # ---- dataset window ----
    num_previous_generations = 4
    train_percent = 1.0
    train_decay = 0.1  # SUBTRACTIVE by design: 100/90/80/70% keeps the window
    # near-flat like a replay buffer, so one bad generation can't steer training
    test_percent = 0.1
    test_decay = 0.01
    dedup_average = True  # merged targets + occurrence sample-weights: kills
    # duplicate-label noise (~70% of samples) without deweighting the opening
    # (the unweighted version cut the empty board from 2.8% of training mass to
    # 0.0027% and the root prior never fit its target again)
    train_batch_size = 1024
    test_batch_size = None


train_config = Connect4Train()


def lr_schedule(generation):
    lr_decay = 0.2 ** (generation // 25)  # 3e-4 -> 6e-5 -> 1.2e-5 -> 2.4e-6 over 100 generations
    return 3e-4 * lr_decay
    # Retuned 2026-07-29 for the MOONSHOT muon scaling (0.2*sqrt(max(d_out,d_in)),
    # muon.py). Production-faithful sweep - real gens 6-9 window with dedup sample
    # weights, warm start from the previous gen's weights, 8 epochs, 3 seeds, best
    # val loss like ModelCheckpoint (tmp/exp_lr_*.log, tmp/results_lr_current_*.json):
    #   moonshot: 1e-4 = 1.302, 3e-4 = 1.237 <- interior minimum, 1e-3 = 1.448, 3e-3 diverges
    #   old rule: best was 3e-3 = 1.316; 1e-2 = 1.496 (cliff, on both rules)
    # Moonshot best-of-basin beats the old rule's by 6% val loss. Under moonshot the
    # per-element step is ~0.2*lr for EVERY layer, so adam-like lrs apply - 3e-4 here
    # plays the role 3e-3 played before. Caveat: measured on one warm-start training
    # step of the loop, not across many generations.


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
