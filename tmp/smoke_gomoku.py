"""End-to-end Gomoku pipeline run in a scratch dir: self play -> train -> onnx -> TRT.

Tiny overrides so it finishes in minutes, but the REAL code path. initial=6 / rest=4 so
the generation-0 split is proven by the data on disk, not just by the helper's return.

Never points at Gomoku/Grok_Zero_Train - that holds 16 generations of real data.
"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

SCRATCH = os.path.join(ROOT, "tmp", "gomoku_smoke", "Grok_Zero_Train")
# Must stay above 1/test_percent or the held-out split rounds to zero games and
# model.fit dies with "The PyDataset has length 0".
INITIAL, REST = 30, 20


def main():
    import h5py as h5
    from pipeline import load_game, Run

    game_class, build_model, configs, real_run_dir = load_game("Gomoku")
    build_config, train_config, optimizer_config = configs
    assert "gomoku_smoke" in SCRATCH and str(real_run_dir) not in SCRATCH

    train_config = train_config.replace(
        initial_games_per_generation=INITIAL,
        games_per_generation=REST,
        total_generations=2,
        num_workers=2,
        MCTS_iteration_limit=20,
        train_batch_size=32,
        test_batch_size=32,
    )
    optimizer_config = dict(optimizer_config)
    optimizer_config["train_epochs"] = 1

    if os.path.exists(os.path.dirname(SCRATCH)):
        shutil.rmtree(os.path.dirname(SCRATCH))
    os.makedirs(SCRATCH, exist_ok=True)

    print(f"scratch run dir: {SCRATCH}")
    print(f"expecting gen0={INITIAL} games, gen1={REST} games\n", flush=True)
    Run(game_class, build_model, (build_config, train_config, optimizer_config), SCRATCH)

    print("\n=== RESULT ===")
    ok = True
    for gen, want in ((0, INITIAL), (1, REST)):
        d = os.path.join(SCRATCH, str(gen))
        h5p = os.path.join(d, "Self_Play_Data.h5")
        if not os.path.exists(h5p):
            print(f"gen {gen}: MISSING {h5p}")
            ok = False
            continue
        with h5.File(h5p, "r") as f:
            n = int(f["game_stats"][2])
        files = sorted(os.listdir(d))
        good = n == want
        ok &= good
        print(f"gen {gen}: {n} games (want {want}) {'OK' if good else 'MISMATCH'} | {files}")
    print("\nSMOKE PASSED" if ok else "\nSMOKE FAILED")


if __name__ == "__main__":
    main()
