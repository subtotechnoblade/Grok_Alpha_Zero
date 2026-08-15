"""Static checks on the Gomoku config: schema, values, and the 10k/5k split."""
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from Gomoku.config import build_config, train_config, optimizer_config, lr_schedule
from Self_Play import games_for_generation

print("=== schema ===")
print(f"type            {type(train_config).__name__}")
print(f"use_gumbel      {train_config['use_gumbel']}  (derived from class)")
print(f"dict bridge     num_previous_generations={train_config['num_previous_generations']}, "
      f"get('dedup_average')={train_config.get('dedup_average')}")

print("\n=== games per generation ===")
for g in (0, 1, 2, 5, 99):
    print(f"  gen {g:>2} -> {games_for_generation(train_config, g)} games")

print("\n=== lr schedule ===")
for g in (0, 19, 20, 39, 40, 99):
    print(f"  gen {g:>2} peak lr {lr_schedule(g):.6f}")

print("\n=== opening book ===")
total = sum(p for _, p in train_config["opening_actions"])
print(f"  {len(train_config['opening_actions'])} forced openings, "
      f"prob sum {total:.3f} ({1-total:.3f} left to the search)")

print("\n=== model builds ===")
from Gomoku.Build_Model import build_model
from Gomoku.game import Gomoku
game = Gomoku()
model = build_model(game.get_input_state().shape, game.policy_shape,
                    (build_config, train_config, optimizer_config))
print(f"  input {game.get_input_state().shape} policy {game.policy_shape} "
      f"params {model.count_params():,}")
names = {l.name for l in model.layers}
for want in optimizer_config["kwargs"]["exclude_layers"] + optimizer_config["orthograd_exclude"]:
    print(f"  exclude_layers {want!r:<12} present in model: {want in names}")

print("\n=== every declared parameter ===")
train_config.explain()
