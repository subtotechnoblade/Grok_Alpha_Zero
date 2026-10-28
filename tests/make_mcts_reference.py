"""Freezes the pre-virtual-loss MCTS behaviour. Run once, from the commit before the change:
python -m tests.make_mcts_reference"""
import os
import pickle
import shutil

from tests.mcts_scenarios import ROOT, SCENARIOS, run_scenario

DATA = os.path.join(ROOT, "tests", "data")

if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "MCTS.py"), os.path.join(DATA, "MCTS_before_vl.py"))
    from MCTS import MCTS

    reference = {}
    for name in SCENARIOS:
        reference[name] = run_scenario(MCTS, name)
        print(f"{name}: {len(reference[name]['records'])} moves", flush=True)
    with open(os.path.join(DATA, "mcts_reference.pkl"), "wb") as f:
        pickle.dump(reference, f)
    print("saved", os.path.join(DATA, "mcts_reference.pkl"))
