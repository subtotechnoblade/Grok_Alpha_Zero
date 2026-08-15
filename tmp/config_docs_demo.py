"""Self-documenting config: help text lives IN the field, everything else is
generated from it. Three views for a beginner, one source of truth:

  1. IDE: ctrl-click / hover on AlphaZeroConfig lands on the field + its help
  2. runtime: cfg.explain() prints every param, its value, default, and help
  3. README: Config.to_markdown() emits a reference table - regenerate on change

Run this file to see views 2 and 3.
"""
from dataclasses import dataclass, field, fields, MISSING


def hp(default=MISSING, *, factory=MISSING, help=""):
    """A documented hyperparameter. `help` is stored in dataclass metadata -
    the sanctioned per-field info channel - so it is machine readable."""
    kw = {"metadata": {"help": help}}
    if factory is not MISSING:
        kw["default_factory"] = factory
    elif default is not MISSING:
        kw["default"] = default
    return field(**kw)


class Documented:
    def explain(self):
        """Print every parameter: value, default, and what it does.
        A star marks values you changed from the default."""
        print(f"{type(self).__name__}")
        print(f"{'':2}{'param':<28} {'value':>10} {'default':>10}  meaning")
        print("  " + "-" * 96)
        for f in fields(self):
            cur = getattr(self, f.name)
            if f.default is not MISSING:
                dflt = f.default
            elif f.default_factory is not MISSING:  # type: ignore
                dflt = f.default_factory()          # type: ignore
            else:
                dflt = "(required)"
            star = "*" if cur != dflt else " "
            print(f" {star}{f.name:<28} {str(cur):>10} {str(dflt):>10}  "
                  f"{f.metadata.get('help', '')}")

    @classmethod
    def to_markdown(cls):
        """README section, generated - docs can never drift from code."""
        lines = [f"### `{cls.__name__}`", "",
                 "| parameter | type | default | meaning |",
                 "|---|---|---|---|"]
        for f in fields(cls):
            if f.default is not MISSING:
                dflt = f"`{f.default}`"
            elif f.default_factory is not MISSING:  # type: ignore
                dflt = f"`{f.default_factory()}`"   # type: ignore
            else:
                dflt = "**required**"
            t = getattr(f.type, "__name__", str(f.type))
            lines.append(f"| `{f.name}` | {t} | {dflt} | "
                         f"{f.metadata.get('help', '')} |")
        return "\n".join(lines)


@dataclass
class TrainConfig(Documented):
    games_per_generation: int = hp(
        help="Self play games per generation. More = better data, slower loop.")
    num_workers: int = hp(
        help="Parallel self play processes. Sweet spot on this machine: 8-12.")
    MCTS_iteration_limit: int = hp(250,
        help="Simulations per move. Rule of thumb: 2-10x the branching factor. "
             "Search depth grows only with log of this.")
    q_weight: float = hp(0.2,
        help="Value target = (1-q)*z + q*rootQ. 0 = pure game outcome (noisy), "
             "1 = pure search value (horizon-limited).")
    create_new_root: bool = hp(True,
        help="Discard the tree between self play moves so policy targets are "
             "fresh sims only. Keep True: reuse polluted targets ~70%.")


@dataclass
class AlphaZeroConfig(TrainConfig):
    c_puct_init: float = hp(1.5,
        help="Exploration width. Higher spreads visits, costing search depth. "
             "Root noise, not this, is the knob for reconsidering moves.")
    dirichlet_alpha: float = hp(1.33,
        help="Root noise concentration, ~10/branching-factor. Only matters "
             "with use_dirichlet on, i.e. during self play.")


if __name__ == "__main__":
    print("=== view 2: what a beginner sees from cfg.explain() ===\n")
    cfg = AlphaZeroConfig(games_per_generation=500, num_workers=10, q_weight=0.25)
    cfg.explain()

    print("\n\n=== view 3: README section from AlphaZeroConfig.to_markdown() ===\n")
    print(AlphaZeroConfig.to_markdown())
