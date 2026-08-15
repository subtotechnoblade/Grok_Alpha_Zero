"""Drive pipeline.Run for one experimental arm into its own scratch run dir.

    python tmp/final_exp/run_arm.py <arm> <total_generations> [workers]

NEVER touches Connect4/Grok_Zero_Train. Each arm gets
tmp/final_exp/<arm>/Grok_Zero_Train/.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# NOTE ON THE BASELINE GAME COUNT.
# HEAD's Connect4/config.py has games_per_generation = 500; the uncommitted
# working tree bumped it to 1000. Every archived run the symptom numbers come
# from (run2, run3) was generated at 500, and the brief defines the scaling arm
# as "500 -> 2000". So 500 is pinned here explicitly for the two 1x arms, and
# the 4x arm is 2000. Nothing else deviates from the working-tree config.
ARMS = {
    # 1. current config, unchanged (game count pinned to 500, see above)
    "baseline": {"games_per_generation": 500},
    # 2. wider search
    "cpuct35": {"games_per_generation": 500, "c_puct_init": 3.5},
    # 3. more data at baseline search width
    "games2000": {"games_per_generation": 2000},
    # 4. both
    "both": {"c_puct_init": 3.5, "games_per_generation": 2000},
    # smoke test only
    "smoke": {"games_per_generation": 20},
}


def main():
    arm = sys.argv[1]
    total_generations = int(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else None

    from Connect4.game import Connect4
    from Connect4 import config as c4
    from pipeline import Run, _LazyBuildModel, RUN_FOLDER_NAME

    overrides = dict(ARMS[arm])
    overrides["total_generations"] = total_generations
    if workers is not None:
        overrides["num_workers"] = workers

    train_config = c4.train_config.replace(**overrides)
    configs = (c4.build_config, train_config, c4.optimizer_config)

    run_dir = ROOT / "tmp" / "final_exp" / arm / RUN_FOLDER_NAME
    run_dir.parent.mkdir(parents=True, exist_ok=True)

    live = (ROOT / "Connect4" / RUN_FOLDER_NAME).resolve()
    assert run_dir.resolve() != live, "refusing to point an arm at the live run"

    print(f"=== ARM {arm} ===")
    print(f"run_dir            {run_dir}")
    print(f"c_puct_init        {train_config['c_puct_init']}")
    print(f"games_per_gen      {train_config['games_per_generation']}")
    print(f"num_workers        {train_config['num_workers']}")
    print(f"total_generations  {train_config['total_generations']}")
    print(f"MCTS_iter_limit    {train_config['MCTS_iteration_limit']}")
    print(f"use_tensorrt       {train_config['use_tensorrt']}")
    sys.stdout.flush()

    Run(Connect4, _LazyBuildModel("Connect4"), configs, run_dir)


if __name__ == "__main__":
    main()
