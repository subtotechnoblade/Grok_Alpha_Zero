"""Prototype D - the template/subclass style the user described.

The base module owns parameter DEFINITIONS (default + help + constraint) and the
primitives (.explain(), .help(), .to_markdown(), .replace(), dict-bridge).
A per-game config file is a child class containing nothing but assignments:

    from Config import AlphaZeroConfig

    class Connect4Train(AlphaZeroConfig):
        games_per_generation = 500
        num_workers = 10
        ...

    train_config = Connect4Train()

Typos and bad values fire AT THE CLASS STATEMENT (import time). Stdlib only.
"""
from dataclasses import MISSING  # reused as a clean sentinel


# ------------------------------------------------------------------ machinery
class Param:
    """One documented hyperparameter: default, help text, optional constraint."""

    __slots__ = ("default", "help", "check")

    def __init__(self, default=MISSING, help="", check=None):
        self.default = default      # MISSING -> required, child must set it
        self.help = help
        self.check = check          # callable(value) -> error string or None


class ConfigBase:
    _params: dict = {}

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        registry = dict(cls.__mro__[1]._params)          # inherit parent's params

        for name, val in list(vars(cls).items()):
            if name.startswith("_") or callable(val) or isinstance(val, (classmethod, staticmethod, property)):
                continue
            if isinstance(val, Param):                    # a template DECLARING a param
                registry[name] = val
                if val.default is not MISSING:
                    setattr(cls, name, val.default)       # plain reads from here on
                continue
            # a child ASSIGNING a value: must be a declared param -> typo firewall
            if name not in registry:
                valid = ", ".join(sorted(registry))
                raise AttributeError(
                    f"{cls.__name__} sets unknown parameter {name!r}. "
                    f"Valid parameters: {valid}")
            err = registry[name].check(val) if registry[name].check else None
            if err:
                raise ValueError(f"{cls.__name__}.{name} = {val!r}: {err}")
        cls._params = registry

    def __init__(self):
        missing = [n for n, p in self._params.items()
                   if p.default is MISSING and isinstance(getattr(type(self), n, MISSING), (Param, type(MISSING)))]
        if missing:
            raise TypeError(f"{type(self).__name__} must set required "
                            f"parameter(s): {', '.join(missing)}")
        self.validate()

    def validate(self):
        """Cross-field checks. Templates extend this."""

    # ---- primitives ----
    @classmethod
    def explain(cls):
        inst = None
        print(f"{cls.__name__}  (* = changed from default)")
        for n, p in cls._params.items():
            cur = getattr(cls, n, "(unset)")
            dflt = "(required)" if p.default is MISSING else p.default
            star = "*" if cur != p.default else " "
            print(f" {star}{n:<28} {str(cur):>10} {str(dflt):>10}  {p.help}")

    @classmethod
    def help(cls, name):
        p = cls._params[name]
        dflt = "(required)" if p.default is MISSING else repr(p.default)
        print(f"{name}: current={getattr(cls, name, '(unset)')!r}, default={dflt}\n  {p.help}")

    @classmethod
    def to_markdown(cls):
        rows = [f"### `{cls.__name__}`", "", "| parameter | default | meaning |", "|---|---|---|"]
        for n, p in cls._params.items():
            d = "**required**" if p.default is MISSING else f"`{p.default}`"
            rows.append(f"| `{n}` | {d} | {p.help} |")
        return "\n".join(rows)

    def replace(self, **overrides):
        """Light copy with validated instance-level overrides - for benchmarks
        that tweak a knob without defining a whole subclass."""
        import copy
        clone = copy.copy(self)
        for k, v in overrides.items():
            if k not in self._params:
                raise AttributeError(f"unknown parameter {k!r}")
            err = self._params[k].check(v) if self._params[k].check else None
            if err:
                raise ValueError(f"{k} = {v!r}: {err}")
            object.__setattr__(clone, k, v)
        clone.validate()
        return clone

    # ---- dict bridge: every legacy cfg["key"] / cfg.get() keeps working ----
    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"{key!r} is not a parameter of {type(self).__name__}")

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __contains__(self, key):
        return hasattr(self, key)


# ------------------------------------------------------------- the templates
def _pos(v):  return None if v > 0 else "must be > 0"
def _frac(v): return None if 0.0 <= v <= 1.0 else "must be in [0, 1]"


class TrainConfig(ConfigBase):
    games_per_generation = Param(help="Self play games per generation.", check=lambda v: None if v >= 2 else "must be >= 2")
    max_actions = Param(help="Hard cap on game length; reaching it without a win is a draw.")
    num_workers = Param(help="Parallel self play processes. Sweet spot here: 8-12.")
    MCTS_iteration_limit = Param(250, help="Simulations per move; depth grows with log of this.", check=_pos)
    q_weight = Param(0.2, help="Value target = (1-q)*z + q*rootQ.", check=_frac)
    create_new_root = Param(True, help="Fresh tree per self play move -> honest policy targets.")
    use_gpu = Param(True, help="GPU for self play inference.")
    use_tensorrt = Param(False, help="TensorRT EP (needs use_gpu).")

    @property
    def use_gumbel(self):
        return isinstance(self, GumbelAZConfig)

    def validate(self):
        if self.use_tensorrt and not self.use_gpu:
            raise ValueError("use_tensorrt=True requires use_gpu=True")


class AlphaZeroConfig(TrainConfig):
    c_puct_init = Param(1.5, help="Exploration width; higher spreads visits, costs depth.", check=_pos)
    dirichlet_alpha = Param(1.33, help="Root noise ~ 10/branching factor.", check=_pos)


class GumbelAZConfig(TrainConfig):
    m = Param(7, help="Actions sampled by sequential halving.", check=_pos)
    c_visit = Param(50.0, help="Gumbel value-scale constant.")
    c_scale = Param(1.0, help="Gumbel value-scale multiplier.")

    def validate(self):
        super().validate()
        s = bin(self.m).count("1")
        need = 2 * self.m - s - (s == 1)
        if self.MCTS_iteration_limit < need:
            raise ValueError(f"m={self.m} needs >= {need + 1} iterations, "
                             f"got {self.MCTS_iteration_limit}")


# ================================================================== the demo
if __name__ == "__main__":
    print("=== a per-game config file is just assignments ===")

    class Connect4Train(AlphaZeroConfig):
        games_per_generation = 500     # connect4 saturates fast, 1000 wastes compute
        max_actions = 42
        num_workers = 10
        use_tensorrt = True
        q_weight = 0.25

    train_config = Connect4Train()
    train_config.explain()

    print("\n=== primitives ===")
    Connect4Train.help("q_weight")
    bench = train_config.replace(num_workers=2)
    print(f"replace(): bench run with num_workers={bench.num_workers}, "
          f"original untouched={train_config.num_workers}")
    print(f"dict bridge: cfg['use_gumbel']={train_config['use_gumbel']}, "
          f"cfg.get('nope', 'dflt')={train_config.get('nope', 'dflt')!r}")

    print("\n=== failure modes, all at the class statement or instantiation ===")
    cases = []

    def try_case(label, fn):
        try:
            fn()
            print(f"   {label:<34} NOT CAUGHT (bad)")
        except (AttributeError, ValueError, TypeError) as e:
            print(f"   {label:<34} {type(e).__name__}: {str(e)[:80]}")

    def typo():
        class T(AlphaZeroConfig):
            games_per_generation = 500; max_actions = 42; num_workers = 10
            create_new_rot = True                       # <- typo

    def badval():
        class T(AlphaZeroConfig):
            games_per_generation = 1; max_actions = 42; num_workers = 10

    def gumbel_key():
        class T(AlphaZeroConfig):
            games_per_generation = 500; max_actions = 42; num_workers = 10
            m = 16                                      # gumbel param on normal AZ

    def missing_required():
        class T(AlphaZeroConfig):
            games_per_generation = 500                  # forgot max_actions etc.
        T()

    def gumbel_iters():
        class T(GumbelAZConfig):
            games_per_generation = 500; max_actions = 42; num_workers = 10
            m = 16; MCTS_iteration_limit = 10
        T()

    try_case("typo'd assignment", typo)
    try_case("bad value", badval)
    try_case("gumbel param on normal AZ", gumbel_key)
    try_case("missing required param", missing_required)
    try_case("gumbel iteration constraint", gumbel_iters)
