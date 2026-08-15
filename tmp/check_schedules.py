"""Print both schedules so the wiring can be eyeballed without a training run."""
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Connect4.config import lr_schedule, optimizer_config, train_config
from Net.Schedules import WarmupCosine

print("GENERATION schedule - peak per generation")
for g in (0, 39, 40, 79, 80, 99):
    print(f"  gen {g:>3}   peak {lr_schedule(g):.6f}")

epochs = optimizer_config["train_epochs"]
# steps/epoch = ceil(unique samples / batch); 10k games deduped to 442,003 -> 432
for games, spe in ((10000, 432), (500, 22)):
    total = spe * epochs
    s = WarmupCosine(lr_schedule(0), total,
                     optimizer_config["lr_warmup_frac"], optimizer_config["lr_floor_frac"])
    print(f"\nSTEP schedule, {games} games: {spe} steps/epoch x {epochs} epochs "
          f"= {total} steps, warmup {s.warmup_steps}")
    for st in sorted({0, 1, s.warmup_steps, total // 4, total // 2, 3 * total // 4, total - 1}):
        print(f"  step {st:>5}  (epoch {st/spe:>5.2f})   lr {float(s(st)):.6f}")
