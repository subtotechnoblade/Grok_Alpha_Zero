"""Typed, self-documenting training configs.

A per-game config file subclasses one of the two algorithm templates and just
assigns values:

    from Config import AlphaZeroConfig

    class Connect4Train(AlphaZeroConfig):
        games_per_generation = 500
        max_actions = 42
        num_workers = 10

    train_config = Connect4Train()

What the template buys over the old plain dict:
  - a typo'd name is an AttributeError AT THE CLASS STATEMENT, not a silent
    .get() default twenty minutes into self play (this exact failure shipped:
    "create_new_root" missing meant tree reuse stayed on unnoticed)
  - a bad value is a ValueError at import, naming the field
  - gumbel parameters cannot exist on a normal AlphaZero config at all - the
    TYPE picks the algorithm, there is no use_gumbel flag to keep in sync
  - .explain() / .help("name") / .to_markdown() are generated from the same
    field definitions, so documentation cannot drift from code

Backward compatibility: instances answer cfg["key"], cfg.get(key, default),
"key" in cfg, and dict(cfg), so every existing call site keeps working.

build_config and optimizer_config deliberately stay plain dicts: they are the
contract between a developer and their own build_model / optimizer, and
standardizing them would only get in the way.

Validation split: VALUE validity (ranges, cross-field rules) lives here and
fires at import. ENVIRONMENT validity (is TensorRT actually installed?) stays
in pipeline.Validate_Configs - config files must stay importable on machines
with no GPU, or self play workers could not load them.
"""
from dataclasses import MISSING  # reused as a clean "no default" sentinel


class Param:
    """One documented hyperparameter: default, help text, optional constraint.

    check is a callable(value) -> None if fine, else an error string.
    default=MISSING marks the parameter required: a game config must set it.
    """

    __slots__ = ("default", "help", "check")

    def __init__(self, default=MISSING, help="", check=None):
        self.default = default
        self.help = help
        self.check = check


def _positive(v):
    return None if v > 0 else "must be > 0"


def _non_negative(v):
    return None if v >= 0 else "must be >= 0"


def _fraction(v):
    return None if 0.0 <= v <= 1.0 else "must be in [0, 1]"


class ConfigBase:
    _params: dict = {}
    _derived: tuple = ()  # property names included in the dict()/keys() view

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        registry = {}
        for base in reversed(cls.__mro__[1:]):  # oldest ancestor first
            registry.update(getattr(base, "_params", {}))

        for name, value in list(vars(cls).items()):
            if (name.startswith("_") or callable(value)
                    or isinstance(value, (classmethod, staticmethod, property))):
                continue
            if isinstance(value, Param):
                # a template DECLARING a parameter
                registry[name] = value
                if value.default is not MISSING:
                    # replace the Param with its default so plain reads work;
                    # required params keep the Param object as a sentinel
                    setattr(cls, name, value.default)
                continue
            # a game config ASSIGNING a value: must name a declared parameter
            if name not in registry:
                raise AttributeError(
                    f"{cls.__name__} sets unknown parameter {name!r}. "
                    f"Valid parameters: {', '.join(sorted(registry))}")
            error = registry[name].check(value) if registry[name].check else None
            if error:
                raise ValueError(f"{cls.__name__}.{name} = {value!r}: {error}")
        cls._params = registry

    def __init__(self):
        missing = [name for name in self._params
                   if isinstance(getattr(type(self), name, None), Param)]
        if missing:
            raise TypeError(f"{type(self).__name__} must set required "
                            f"parameter(s): {', '.join(missing)}")
        self.validate()

    def validate(self):
        """Cross-field rules. Templates extend this and call super()."""

    # ------------------------------------------------------------- primitives
    def explain(self):
        """Every parameter: value, default, meaning. * marks overrides."""
        print(f"{type(self).__name__}  (* = changed from default)")
        for name, p in self._params.items():
            current = getattr(self, name)
            default = "(required)" if p.default is MISSING else p.default
            star = "*" if current != p.default else " "
            print(f" {star}{name:<28} {str(current):>12} {str(default):>12}  {p.help}")

    @classmethod
    def help(cls, name):
        """Print one parameter's documentation."""
        p = cls._params[name]
        default = "(required)" if p.default is MISSING else repr(p.default)
        current = getattr(cls, name, "(unset)")
        print(f"{name}: current={current!r}, default={default}\n  {p.help}")

    @classmethod
    def to_markdown(cls):
        """README reference table, generated so it cannot drift from the code."""
        lines = [f"### `{cls.__name__}`", "",
                 "| parameter | default | meaning |", "|---|---|---|"]
        for name, p in cls._params.items():
            default = "**required**" if p.default is MISSING else f"`{p.default}`"
            lines.append(f"| `{name}` | {default} | {p.help} |")
        return "\n".join(lines)

    def replace(self, **overrides):
        """Copy with validated overrides - for benchmarks and one-off runs."""
        import copy
        clone = copy.copy(self)
        for key, value in overrides.items():
            if key not in self._params:
                raise AttributeError(f"unknown parameter {key!r}")
            error = self._params[key].check(value) if self._params[key].check else None
            if error:
                raise ValueError(f"{key} = {value!r}: {error}")
            object.__setattr__(clone, key, value)
        clone.validate()
        return clone

    # ------------------------------------------ dict bridge (backward compat)
    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"{key!r} is not a parameter of {type(self).__name__}")

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __contains__(self, key):
        return hasattr(self, key)

    def keys(self):
        # _derived makes dict(cfg) carry computed values like use_gumbel, so a
        # plain-dict copy handed to Self_Play still answers cfg["use_gumbel"]
        return list(self._params) + list(self._derived)


class TrainConfig(ConfigBase):
    """Everything shared between normal and gumbel AlphaZero. Not usable
    directly - subclass AlphaZeroConfig or GumbelAZConfig."""

    _derived = ("use_gumbel",)

    # ---- the training loop ----
    total_generations = Param(100, help=(
        "Generations to train. One generation = self play -> train -> export. "
        "Training can be stopped and resumed at any point."), check=_positive)
    games_per_generation = Param(help=(
        "Self play games per generation. More = better data, slower loop."),
        check=lambda v: None if v >= 2 else "must be >= 2")
    initial_games_per_generation = Param(None, help=(
        "Games for generation 0 only, which has no network and is the base every "
        "later generation bootstraps from. None = use games_per_generation."),
        check=lambda v: None if v is None or v >= 2 else "must be >= 2 or None")
    max_actions = Param(help=(
        "Hard cap on game length. Reaching it WITHOUT a win is scored a draw."),
        check=_positive)

    # ---- self play workers / inference ----
    num_workers = Param(help=(
        "Parallel self play processes. They share one inference server."),
        check=_positive)
    use_gpu = Param(True, help="Run self play inference on the GPU.")
    use_tensorrt = Param(False, help=(
        "TensorRT execution provider (needs use_gpu, an NVIDIA card, and the "
        "tensorrt requirements file). ~45-120s engine build per generation."))
    trt_fp16 = Param(True, help=(
        "Let TensorRT pick fp16 kernels. Inference only - independent of the "
        "TRAINING precision in build_config[\"mixed_precision\"]."))
    use_inference_server = Param(True, help=(
        "Batch all workers' requests through one server process. Keep True "
        "unless debugging - per-worker sessions duplicate GPU memory."))
    max_cache_depth = Param(15, help=(
        "Memoise network outputs for positions shallower than this ply. Only "
        "the opening region repeats across games, deep entries never hit."),
        check=_non_negative)
    dml_device_id = Param(None, help=(
        "Pin the DirectML adapter on multi-GPU machines (None = default). "
        "On this machine: 0 = 760M igpu, 1 = RTX 2060."))

    # ---- search ----
    MCTS_iteration_limit = Param(250, help=(
        "Simulations per move. Depth grows only with the LOG of this, at the "
        "effective branching factor the policy leaves open."), check=_positive)
    MCTS_time_limit = Param(None, help=(
        "Seconds per move instead of a sim count. Leave None for training - "
        "variable sim counts make inconsistent policy targets."))
    use_njit = Param(None, help=(
        "Force numba on/off for the search kernels. None = on."))
    create_new_root = Param(True, help=(
        "Discard the tree between SELF PLAY moves so the recorded policy "
        "target is exactly this position's fresh sims under this dirichlet "
        "draw. With reuse, ~50-70% of the target was inherited visits "
        "sharpened toward the incumbent move. Play-time keeps reuse: there "
        "visits are compute, not a training label."))

    # ---- training targets ----
    q_weight = Param(0.2, help=(
        "Value target = (1-q)*z + q*rootQ. 0 = pure game outcome (unbiased, "
        "noisy), 1 = pure search value (suffers the horizon effect)."),
        check=_fraction)
    opening_actions = Param([], help=(
        "Forced-opening book: [[action, prob], ...] applied to move 0. "
        "Probabilities may sum below 1, the remainder plays the search move. "
        "Partial books re-falsify beliefs the search refuses to test; a 100% "
        "book previously produced the always-draw network - keep it partial."))
    num_explore_actions_first = Param(3, help=(
        "Plies sampled at tau=1 for the FIRST player (its first N moves). "
        "Nearly the only entropy source in self play - root noise cannot "
        "overturn a value gap at practical sim counts."), check=_non_negative)
    num_explore_actions_second = Param(3, help=(
        "Same for the second player."), check=_non_negative)

    # ---- dataset window ----
    num_previous_generations = Param(4, help=(
        "How many past generations join the training pool (replay buffer)."),
        check=_non_negative)
    train_percent = Param(1.0, help=(
        "Fraction of the newest generation's games used for training."),
        check=_fraction)
    train_decay = Param(0.1, help=(
        "SUBTRACTED from train_percent per generation back (0.1 -> 100%, 90%, "
        "80%...). Deliberately gentle: the near-flat window is a replay buffer "
        "that stops one bad generation from steering training."), check=_fraction)
    test_percent = Param(0.1, help=(
        "Fraction of games held out for validation, split BY GAME so a "
        "position and its mirror can't straddle the split."), check=_fraction)
    test_decay = Param(0.01, help=(
        "Subtracted from test_percent per generation back."), check=_fraction)
    dedup_average = Param(True, help=(
        "Merge byte-identical states; targets become the occurrence-weighted "
        "mean and the occurrence count becomes the sample weight. Removes "
        "duplicate-label noise without deweighting the opening."))
    max_sample_weight = Param(None, help=(
        "Split any deduped train sample heavier than this into equal copies spread "
        "across batches. Same objective, without one position dominating a batch. "
        "None = off."), check=lambda v: None if v is None or v >= 1 else "must be >= 1 or None")
    train_batch_size = Param(1024, help="Samples per training batch.",
                             check=_positive)
    test_batch_size = Param(None, help="None = use train_batch_size.")

    @property
    def use_gumbel(self):
        """Derived from the class, not stored - there is no flag to forget."""
        return isinstance(self, GumbelAZConfig)

    def validate(self):
        if self.use_tensorrt and not self.use_gpu:
            raise ValueError("use_tensorrt=True requires use_gpu=True")
        if self.opening_actions:
            total = sum(prob for _, prob in self.opening_actions)
            if total > 1.0 + 1e-9:
                raise ValueError(
                    f"opening_actions probabilities sum to {total:.3f} > 1")


class AlphaZeroConfig(TrainConfig):
    """Classic AlphaZero self play: PUCT + root dirichlet noise + tau sampling."""

    c_puct_init = Param(1.5, help=(
        "Exploration width. Higher spreads visits across siblings, which "
        "costs search depth. Root dirichlet noise, not this, is the "
        "mechanism for reconsidering a move the prior dismissed."),
        check=_positive)
    dirichlet_alpha = Param(1.33, help=(
        "Root noise concentration, rule of thumb 10 / branching factor."),
        check=_positive)
    dirichlet_epsilon = Param(0.25, help=(
        "Noise mixing weight at the root: (1-eps)*prior + eps*Dir(alpha)."),
        check=_fraction)


class GumbelAZConfig(TrainConfig):
    """Gumbel AlphaZero (Danihelka et al. 2022): sequential halving with "
    "gumbel noise instead of PUCT + dirichlet. Far fewer sims needed."""

    m = Param(7, help=(
        "Actions sampled in the first stage of sequential halving."),
        check=_positive)
    c_visit = Param(50.0, help="Value-scale constant in the gumbel move mix.")
    c_scale = Param(1.0, help="Multiplier on the value scale.")

    def validate(self):
        super().validate()
        if self.MCTS_time_limit is not None:
            raise ValueError(
                "gumbel uses iteration_limit for sequential halving, "
                "MCTS_time_limit must be None")
        ones = bin(self.m).count("1")
        min_iterations = 2 * self.m - ones - (ones == 1)
        if self.MCTS_iteration_limit < min_iterations:
            raise ValueError(
                f"sequential halving with m={self.m} needs at least "
                f"{min_iterations + 1} iterations, got {self.MCTS_iteration_limit}")


if __name__ == "__main__":
    # regenerate the README reference:  python Config.py > docs/CONFIG.md
    print(AlphaZeroConfig.to_markdown())
    print()
    print(GumbelAZConfig.to_markdown())
