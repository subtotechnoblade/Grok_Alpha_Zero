"""Three prototypes for a typed train config. Run this file to see each one's
failure modes fire. No new dependencies - stdlib only.

  A: dataclass inheritance   - AlphaZeroConfig / GumbelAZConfig subclass a base
  B: dataclass composition   - TrainConfig has a .search sub-object per variant
  C: TypedDict               - annotations only, zero runtime behavior change

build_config deliberately stays a plain dict in all three: it is the contract
between a developer and THEIR OWN build_model, so standardizing it would just
get in their way.
"""
from dataclasses import dataclass, field, fields
from typing import Optional, Union


# ------------------------------------------------------------------ shared bits
class DictBridge:
    """Backward compatibility: every existing `cfg["key"]` / `cfg.get("key", d)`
    / `"key" in cfg` in Self_Play, MCTS, pipeline, Dataloader keeps working
    unchanged. New code uses attributes and gets IDE autocomplete. Migrate call
    sites whenever you happen to touch them - no big-bang rewrite."""

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"{key!r} is not a field of {type(self).__name__}. "
                           f"Valid fields: {[f.name for f in fields(self)]}")

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __contains__(self, key):
        return hasattr(self, key)


# =============================================================================
# PROTOTYPE A - inheritance. The TYPE is the algorithm choice.
# =============================================================================
@dataclass
class TrainConfigA(DictBridge):
    # ---- required: no defaults, forgetting one is an immediate TypeError ----
    games_per_generation: int
    max_actions: int
    num_workers: int
    MCTS_iteration_limit: int

    # ---- shared knobs, defaults = current connect4 values ----
    total_generations: int = 100
    num_explore_actions_first: int = 3
    num_explore_actions_second: int = 3
    use_gpu: bool = True
    use_tensorrt: bool = False
    trt_fp16: bool = True
    use_inference_server: bool = True
    max_cache_depth: int = 15
    MCTS_time_limit: Optional[float] = None
    use_njit: Optional[bool] = None            # None = infer per platform
    create_new_root: bool = True
    q_weight: float = 0.2
    opening_actions: list = field(default_factory=list)
    num_previous_generations: int = 4
    train_percent: float = 1.0
    train_decay: float = 0.1                   # subtractive, per Brian's design
    test_percent: float = 0.1
    test_decay: float = 0.01
    dedup_average: bool = True
    train_batch_size: int = 1024
    test_batch_size: Optional[int] = None
    dml_device_id: Optional[int] = None

    @property
    def use_gumbel(self) -> bool:
        # derived from the type - there is no flag to keep in sync, and the
        # dict bridge makes existing `train_config["use_gumbel"]` reads hit this
        return isinstance(self, GumbelAZConfigA)

    def __post_init__(self):
        # value validity lives HERE and fires at `import config`.
        # environment validity (is TensorRT installed?) stays in pipeline.py -
        # the config file must stay importable on machines without a GPU.
        if self.total_generations <= 0:
            raise ValueError("total_generations must be > 0")
        if self.games_per_generation < 2:
            raise ValueError("games_per_generation must be >= 2")
        if self.use_tensorrt and not self.use_gpu:
            raise ValueError("use_tensorrt=True requires use_gpu=True")
        if not 0.0 <= self.q_weight <= 1.0:
            raise ValueError(f"q_weight must be in [0, 1], got {self.q_weight}")
        if self.opening_actions:
            total = sum(p for _, p in self.opening_actions)
            if total > 1.0 + 1e-9:
                raise ValueError(f"opening_actions probabilities sum to {total} > 1")


@dataclass
class AlphaZeroConfigA(TrainConfigA):
    """Classic AlphaZero: PUCT + dirichlet root noise + tau sampling."""
    c_puct_init: float = 1.5
    dirichlet_alpha: float = 1.33
    dirichlet_epsilon: float = 0.25

    def __post_init__(self):
        super().__post_init__()
        if self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be > 0")


@dataclass
class GumbelAZConfigA(TrainConfigA):
    """Gumbel AlphaZero (Danihelka et al.): sequential halving, no dirichlet."""
    m: int = 7
    c_visit: float = 50.0
    c_scale: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.MCTS_time_limit is not None:
            raise ValueError("gumbel needs iteration_limit, not time_limit")
        s = bin(self.m).count("1")
        min_iters = 2 * self.m - s - (s == 1)
        if self.MCTS_iteration_limit < min_iters:
            raise ValueError(
                f"sequential halving with m={self.m} needs >= {min_iters + 1} "
                f"MCTS iterations, got {self.MCTS_iteration_limit}")


# =============================================================================
# PROTOTYPE B - composition. The algorithm is a sub-object on one config class.
# =============================================================================
@dataclass
class PUCTSearch:
    c_puct_init: float = 1.5
    dirichlet_alpha: float = 1.33
    dirichlet_epsilon: float = 0.25


@dataclass
class GumbelSearch:
    m: int = 7
    c_visit: float = 50.0
    c_scale: float = 1.0


@dataclass
class TrainConfigB(DictBridge):
    search: Union[PUCTSearch, GumbelSearch]
    games_per_generation: int = 500
    max_actions: int = 42
    num_workers: int = 10
    MCTS_iteration_limit: int = 250
    # ... same shared fields as A ...

    @property
    def use_gumbel(self) -> bool:
        return isinstance(self.search, GumbelSearch)

    def get(self, key, default=None):
        # flatten lookups into the search object too, so legacy
        # train_config["c_puct_init"] / ["m"] reads keep working
        if hasattr(self, key):
            return getattr(self, key)
        return getattr(self.search, key, default)

    def __getitem__(self, key):
        v = self.get(key, _MISSING := object())
        if v is _MISSING:
            raise KeyError(key)
        return v


# =============================================================================
# PROTOTYPE C - TypedDict. Static hints only, dicts stay dicts.
# =============================================================================
from typing import TypedDict


class TrainConfigC(TypedDict, total=False):
    games_per_generation: int
    max_actions: int
    num_workers: int
    MCTS_iteration_limit: int
    c_puct_init: float
    dirichlet_alpha: float
    q_weight: float
    # ... etc. PyCharm flags typos and wrong types AT EDIT TIME, but nothing
    # is checked at runtime and gumbel/normal separation is not enforceable.


# =============================================================================
# demo: what each failure mode looks like with prototype A
# =============================================================================
if __name__ == "__main__":
    print("== valid config constructs fine ==")
    cfg = AlphaZeroConfigA(games_per_generation=500, max_actions=42,
                           num_workers=10, MCTS_iteration_limit=250,
                           use_tensorrt=True)
    print(f"   type={type(cfg).__name__}  use_gumbel={cfg.use_gumbel}")
    print(f"   old-style reads still work: cfg['q_weight']={cfg['q_weight']}, "
          f"cfg.get('missing', 'dflt')={cfg.get('missing', 'dflt')!r}")
    import pickle
    print(f"   pickles for mp spawn: {pickle.loads(pickle.dumps(cfg)).num_workers=}")

    print("\n== failure modes, all at import time ==")
    for label, thunk in [
        ("typo'd key", lambda: AlphaZeroConfigA(games_per_generation=500,
            max_actions=42, num_workers=10, MCTS_iteration_limit=250,
            create_new_rot=True)),
        ("bad value", lambda: AlphaZeroConfigA(games_per_generation=1,
            max_actions=42, num_workers=10, MCTS_iteration_limit=250)),
        ("gumbel key on normal AZ", lambda: AlphaZeroConfigA(
            games_per_generation=500, max_actions=42, num_workers=10,
            MCTS_iteration_limit=250, m=16)),
        ("too few sims for gumbel m", lambda: GumbelAZConfigA(
            games_per_generation=500, max_actions=42, num_workers=10,
            MCTS_iteration_limit=10, m=16)),
        ("misspelled read", lambda: AlphaZeroConfigA(games_per_generation=500,
            max_actions=42, num_workers=10,
            MCTS_iteration_limit=250)["q_weigth"]),
    ]:
        try:
            thunk()
            print(f"   {label:<28} NOT CAUGHT (bad)")
        except (TypeError, ValueError, KeyError) as e:
            msg = str(e).replace("\n", " ")[:95]
            print(f"   {label:<28} {type(e).__name__}: {msg}")
