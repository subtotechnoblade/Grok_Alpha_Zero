import os
import time
from pathlib import Path

import h5py as h5
import numpy as np

from tqdm import tqdm
from warnings import warn

import multiprocessing as mp

from MCTS import MCTS
from MCTS_Gumbel import MCTS_Gumbel

from Client_Server import (Parallelized_Session, start_server, convert_to_single_info,
                           create_shared_memory, create_events)


class Self_Play:
    def __init__(self,
                 game,
                 sess,
                 build_config: dict,
                 train_config: dict,
                 lock: mp.Lock,
                 folder_path: str,
                 generation: int):

        self.game = game
        self.sess = sess
        self.build_config = build_config
        self.train_config = train_config
        self.lock = lock
        self.folder_path = folder_path
        self.generation = generation

        self.iteration_limit = self.train_config["MCTS_iteration_limit"]
        self.time_limit = self.train_config["MCTS_time_limit"]

        if train_config["use_gumbel"] is False:
            dirichlet_epsilon = 0.25
            self.mcts: MCTS = MCTS(game=self.game,
                                   session=self.sess if self.generation > 0 else None,
                                   use_njit=train_config["use_njit"],
                                   c_puct_init=self.train_config["c_puct_init"],
                                   use_dirichlet=True,
                                   dirichlet_alpha=self.train_config["dirichlet_alpha"],
                                   dirichlet_epsilon=dirichlet_epsilon,
                                   tau=1.0,
                                   fast_find_win=False)
        else:
            self.mcts_gumbel = self.create_MCTS_Gumbel()

    def create_MCTS_Gumbel(self):
        return MCTS_Gumbel(game=self.game,
                                 session=self.sess,
                                 use_gumbel_noise=True,
                                 use_njit=self.train_config["use_njit"],
                                 m=self.train_config["m"],
                                 c_visit=self.train_config["c_visit"],
                                 c_scale=self.train_config["c_scale"],
                                 activation_fn="stablemax" if self.build_config.get("use_stablemax") else "softmax")

    def play(self):
        board_states = []
        improved_policies = []
        target_q = []
        target_z = []

        actions_count = 0
        winner = -2
        while winner == -2 and actions_count < self.train_config["max_actions"]:
            board_states.append(self.game.get_input_state().copy())

            current_move_num = len(self.game.action_history)

            # using normal MCTS
            if not self.train_config["use_gumbel"]:
                if current_move_num % 2 == 0:
                    explore = current_move_num // 2 < self.train_config["num_explore_actions_first"]
                else:
                    explore = (current_move_num - 1) // 2 < self.train_config["num_explore_actions_second"]
                self.mcts.update_hyperparams(tau=1.0 if explore else 0)

                action, action_probs = self.mcts.run(
                    iteration_limit=int(self.iteration_limit * 1.5),
                    time_limit=self.time_limit,
                    use_bar=False)

            # else if we are using gumbel we need to reset the tree class
            else:
                action, action_probs = self.mcts_gumbel.run(
                    iteration_limit=self.iteration_limit,
                    time_limit=self.time_limit, use_bar=False)

            improved_policy = self.game.compute_policy_improvement(map(lambda x: x[:2], action_probs)) # This takes the first and seconds element of which is the [action, prob]
            improved_policies.append(improved_policy)

            if np.array_equal(action_probs[0][0], action):
                target_q.append(action_probs[0][2])
            else:
                # we need to find the q assigned for the played action (because gumbel is a little weird, the 0th action in action probs
                # isn't always the one that gets played)
                for i in range(1, len(action_probs)):
                    if np.array_equal(action_probs[i][0], action):
                        target_q.append(action_probs[i][2])
                        break

            target_z.append(self.game.get_next_player())  # Important that this is before do_action()
            # We can safely say that target_values are the players that played the move, not the next player

            if current_move_num == 0 and self.train_config.get("opening_actions", False):
                sample_actions, weights = zip(*self.train_config["opening_actions"])
                sample_actions, weights = list(sample_actions), list(weights)
                sum_weights = sum(weights)
                if sum_weights < 1.0:
                    sample_actions.append(action)
                    weights.append(1.0 - sum_weights)
                sample_actions = np.array(sample_actions)

                idx = np.random.choice(len(sample_actions), size=1, p=weights, replace=False)[0]
                action = np.array(sample_actions[idx])

            self.game.do_action(action)

            winner = self.game.check_win()

            if winner == -2:  # or else there will be an error, because you are pruning a winning move
                if not self.train_config["use_gumbel"]:
                    create_new_root = self.train_config.get("create_new_root", False)
                    self.mcts.prune_tree(action, create_new_root)
                else:
                    del self.mcts_gumbel
                    self.mcts_gumbel = self.create_MCTS_Gumbel()

            actions_count += 1
            if actions_count == self.train_config["max_actions"] and winner == -2:
                winner = 0 # simple cutoff in case the games get way too long.
                # winner == -2 guard: check_win already ran this ply, and a game can be
                # WON on the max_actions-th move. The unguarded version overwrote those
                # wins to draws - 188 games across gens 0-9, all stolen from player +1
                # (the mover on the final ply), zeroing their value targets and
                # inflating every draw rate in game_stats

        board_states = np.array(board_states, dtype=self.game.board.dtype)
        improved_policies = np.array(improved_policies, dtype=np.float32)

        target_q = np.array(target_q, dtype=np.float32).reshape((-1, 1))
        target_z = np.array(target_z, dtype=np.float32).reshape((-1, 1))

        if winner == target_z[-1][0] == -1:  # if player -1 just won
            target_z *= -1.0  # Flip it so that the player that won, evaluates to 1 (winner)
        elif winner == 0:  # if it a draw or
            target_z[:] = 0.0
        # else the player that played was 1, and won which is 1, thus no need to invert
        # augmentation

        # weighted average of q and z. q alone suffers from the horizon effect, z alone is
        # too noisy; q_weight = 0.5 is the even average from the oracle blog post
        q_weight = self.train_config.get("q_weight", 0.5)
        if self.generation == 0:
            # Generation 0 plays with NO network (session is None, see self_play_task), so
            # every non-terminal leaf evaluates to exactly 0.0 and rootQ carries only
            # whatever terminal outcomes the tree happened to reach. Blending that in does
            # not smooth z, it drags the target toward zero by a factor the later
            # generations never see. Warm up on the pure game outcome instead.
            q_weight = 0.0
        target_values = (1.0 - q_weight) * target_z + q_weight * target_q

        augmented_board_states, augmented_policies = self.game.augment_sample(board_states, improved_policies)
        augmented_values = np.repeat(np.expand_dims(target_values, 0), repeats=augmented_policies.shape[0], axis=0)

        # Assume that a .h5 file has been created and the max moves dataset is already created
        with self.lock, h5.File(f"{self.folder_path}/Self_Play_Data.h5", "r+") as file:
            game_length = len(self.game.action_history)

            # [max_actions, total_actions, num_unaugmented_games, player -1 wins, draws, player 1 wins]
            if file["game_stats"][0] < game_length:
                file["game_stats"][0] = game_length

            file["game_stats"][1] += board_states.shape[0]  # adds to
            file["game_stats"][2] += 1  # adds a game to num_unaugmented_games

            file["game_stats"][winner + 4] += 1  # adding winners/ draws

            dataset_name = (len(file.keys()) - 1) // 3  # starts from 0
            for inc in range(augmented_policies.shape[0]):
                file.create_dataset(f"boards_{dataset_name + inc}",
                                    maxshape=(None, *augmented_board_states[inc].shape[1:]),
                                    dtype=augmented_board_states[inc].dtype,
                                    data=augmented_board_states[inc],
                                    chunks=None)

                file.create_dataset(f"policies_{dataset_name + inc}",
                                    maxshape=(None, *augmented_policies[inc].shape[1:]),
                                    dtype=np.float32,
                                    data=augmented_policies[inc],
                                    chunks=None)

                file.create_dataset(f"values_{dataset_name + inc}",
                                    maxshape=(None, *augmented_values[inc].shape[1:]),
                                    dtype=np.float32,
                                    data=augmented_values[inc],
                                    chunks=None)


def self_play_task(worker_id,
                   info,
                   game_class,
                   input_feed_info: dict,
                   output_feed_info: dict,
                   build_config: dict,
                   train_config: dict,
                   lock: mp.Lock,
                   folder_path: str,
                   generation: int,
                   claimed,
                   total_games: int,
                   progress,
                   stats_q,
                   request_ready=None,
                   response_ready=None):
    # This worker lives for the whole generation and pulls games until the shared counter
    # runs out. Everything before the loop is paid ONCE instead of once per game: the
    # process spawn, the session, the numba warm up, and the memo below.
    np.random.seed()  # once per PROCESS. Never per game: pointless at best, and a fixed
    # value here would correlate every worker's games with each other.
    import onnxruntime as rt
    session = None
    if generation > 0:
        if train_config["use_inference_server"]:
            session = Parallelized_Session(worker_id,
                                           info,
                                           input_feed_info,
                                           output_feed_info,
                                           request_ready,
                                           response_ready, )
        else:
            providers, onnx_path = info
            session = rt.InferenceSession(onnx_path, providers=providers)

        if train_config.get("max_cache_depth", 0) not in [None, 0]: # so both 0 and None are excluded
            from Session_Cache import Cache_Wrapper
            session = Cache_Wrapper(session, train_config["max_cache_depth"])
            # Wrapped ONCE here, so this single memo is shared by every game this worker
            # plays AND by the whole search inside each game (Self_Play hands the same
            # session object to both trees). That sharing is the point, not an accident.

    while True:
        # Claim one game. The read and the increment must happen under the SAME lock
        # acquisition, otherwise two workers can both see the last index and the
        # generation overshoots total_games.
        with claimed.get_lock():
            if claimed.value >= total_games:
                break
            claimed.value += 1

        Self_Play(game_class(),
                  session,
                  build_config,
                  train_config,
                  lock,
                  folder_path,
                  generation).play()
        # a fresh Self_Play per game is intended, it owns self.game and mutates it
        with progress.get_lock():
            progress.value += 1

    # Hand the memo stats to the parent rather than printing here: the tqdm bar is live
    # in the parent and its \r redraws overwrite anything a worker writes to the terminal.
    # Always put exactly one item so the parent knows how many to expect.
    if generation > 0 and train_config.get("max_cache_depth", 0) not in [None, 0]:
        stats_q.put((worker_id, session.stats()))
    else:
        stats_q.put((worker_id, None))

    if train_config["use_inference_server"]:
        info.close()


def convert_shape(shape):
    assert len(shape) > 0
    str_shape = ""
    for dim in shape:
        str_shape += f"{dim}x"
    return str_shape[:-1]


def games_for_generation(train_config, generation):
    """Generation 0 may run a larger book than the rest. .get keeps plain-dict configs working."""
    initial = train_config.get("initial_games_per_generation")
    if generation == 0 and initial:
        return int(initial)
    return int(train_config["games_per_generation"])


def run_self_play(game_class,
                  configs,
                  folder_path,
                  per_process_wait_time=1e-3):
    build_config, train_config, optimizer_config = configs
    if not os.path.exists(f"{folder_path}/Self_Play_Data.h5"):
        raise ValueError("Dataset file hasn't been created. Self play depends on that file!")

    generation = int(Path(folder_path).name)  # separator agnostic, see Dataloader
    target_games = games_for_generation(train_config, generation)

    with h5.File(f"{folder_path}/Self_Play_Data.h5") as dataset_file:
        num_games_left = int(target_games - dataset_file["game_stats"][2])

    if num_games_left <= 0:
        print(f"Finished generating {target_games} games!")
        return

    num_workers = train_config["num_workers"]
    if num_games_left < num_workers:
        num_workers = num_games_left

    game = game_class()
    inputs_shape = game.get_input_state().shape
    policy_shape = game.policy_shape
    str_board_shape = convert_shape(inputs_shape)
    del game

    import onnxruntime as rt
    available_providers = rt.get_available_providers()
    onnx_file_path = f"{folder_path}/model.onnx"
    if train_config["use_gpu"]:
        if train_config["use_tensorrt"]:
            max_shape = train_config["num_workers"]
            onnx_file_path = f"{folder_path}/TRT_cache/model_ctx.onnx"

            providers = [
                ('TensorrtExecutionProvider', {
                    "trt_engine_cache_enable": True,
                    "trt_dump_ep_context_model": True,
                    "trt_builder_optimization_level": 5,
                    "trt_auxiliary_streams": 0,

                    "trt_ep_context_file_path": f"{folder_path}/TRT_cache/",
                    "trt_profile_min_shapes": f"inputs:1x{str_board_shape}",
                    "trt_profile_max_shapes": f"inputs:{max_shape}x{str_board_shape}",
                    "trt_profile_opt_shapes": f"inputs:{max_shape}x{str_board_shape}",
                }),
                'CUDAExecutionProvider',
                'CPUExecutionProvider']
            # inference precision is independent of the TRAINING precision policy. falling
            # back to the old mixed_precision gate keeps games without the key unchanged
            if train_config.get("trt_fp16",
                                build_config.get("mixed_precision") == "mixed_float16"):
                providers[0][1]["trt_fp16_enable"] = True
        elif "CUDAExecutionProvider" in available_providers:
            providers = [
                'CUDAExecutionProvider',
                'CPUExecutionProvider']
        elif "DmlExecutionProvider" in available_providers:
            # on a machine with more than one DX12 adapter, DirectML picks one for you.
            # measured here: device 0 is the 760M igpu, device 1 is the rtx 2060, and the
            # default lands on the 2060. Set dml_device_id in train_config to pin it
            dml_device_id = train_config.get("dml_device_id")
            providers = [
                ('DmlExecutionProvider', {"device_id": dml_device_id})
                if dml_device_id is not None else 'DmlExecutionProvider',
                'CPUExecutionProvider']
        else:
            raise RuntimeError("No GPU execution providers available")
    else:
        providers = ['CPUExecutionProvider']

    batched_input_feed_info = {"inputs": [[-1, *inputs_shape], np.float32]}
    batched_output_feed_info = {"policy": [-1, *policy_shape],
                                "value": [-1, 1]}

    print(
        f"Running with {num_workers} workers for {num_games_left} games with {onnx_file_path} for generation: {generation}!\n")
    bar = tqdm(total=num_games_left, desc="Generating self play games")
    shms = []
    request_ready, response_events = None, [None] * num_workers
    if train_config["use_inference_server"]:
        shms = create_shared_memory(batched_input_feed_info, batched_output_feed_info, num_workers)
        request_ready, response_events = create_events(num_workers)
        server = mp.Process(target=start_server, args=(batched_input_feed_info,
                                                       batched_output_feed_info,
                                                       shms,
                                                       providers,
                                                       onnx_file_path,
                                                       per_process_wait_time,
                                                       request_ready,
                                                       response_events))
        server.start()

    lock = mp.Lock()
    progress = mp.Value("i", 0)
    stats_q = mp.Queue()

    # Games are claimed one at a time from this counter instead of being split up front.
    # A static split makes the generation as long as its SLOWEST worker's share: whoever
    # finishes early exits and its cores idle while the stragglers grind out the tail.
    # Pulling on demand self-corrects for any cause of imbalance (slow core, unlucky run
    # of long games, worse memo hit rate) without needing to know which one it was, and
    # bounds the tail at one game instead of one share.
    #
    # This does NOT disturb what the old comment here protected: workers still spawn once
    # and live the whole generation, so worker i still owns shm slot i for its entire
    # life, which is the invariant Parallelized_Session wants. Only the loop bound moved.
    claimed = mp.Value("i", 0)

    jobs = []
    for worker_id in range(num_workers):
        worker = mp.Process(target=self_play_task,
                            args=(worker_id,
                                  shms[worker_id] if train_config["use_inference_server"] else [providers,
                                                                                                onnx_file_path],
                                  game_class,
                                  convert_to_single_info(batched_input_feed_info),
                                  convert_to_single_info(batched_output_feed_info),
                                  build_config,
                                  train_config,
                                  lock,
                                  folder_path,
                                  generation,
                                  claimed,
                                  num_games_left,
                                  progress,
                                  stats_q,
                                  request_ready,
                                  response_events[worker_id]),
                            name=f"{worker_id}")
        worker.start()
        jobs.append(worker)

    last = 0
    while any(worker.is_alive() for worker in jobs):
        time.sleep(0.25)  # the old loop polled is_alive() with no sleep, pinning a core
        current = progress.value
        if current > last:
            bar.update(current - last)
            last = current

        if train_config["use_inference_server"] and not server.is_alive():
            # Without this the run hangs forever: Parallelized_Session.run spins on the
            # shared memory flag with no timeout, so a dead server leaves every worker
            # burning a core while still reporting is_alive() == True.
            for worker in jobs:
                worker.terminate()
            for shm in shms:
                shm.unlink()
            raise RuntimeError("Inference server died mid generation, self play cannot continue")
    bar.update(progress.value - last)
    bar.close()

    # Drain before joining: a process that has queued items does not exit until they are
    # flushed, and the bar is closed now so these lines will not be overwritten.
    memo_stats = []
    for _ in range(len(jobs)):
        try:
            worker_id, s = stats_q.get(timeout=10)
            if s is not None:
                memo_stats.append((worker_id, s))
        except Exception:
            break  # a crashed worker never reported, the exitcode check below handles it
    for worker_id, s in sorted(memo_stats):
        print(f"worker {worker_id}: memo {s['hits']} hits / {s['misses']} misses "
              f"({s['hit_rate']:.1%} hit rate), {s['entries']} entries")

    for worker in jobs:
        worker.join()
        if worker.exitcode != 0:
            if train_config["use_inference_server"]:
                server.terminate()
            for shm in shms:
                shm.unlink()
            raise RuntimeError("Main process stopped as self play worker ran into an error")

    if train_config["use_inference_server"]:
        server.terminate()
        for shm in shms:
            shm.unlink()

    with h5.File(f"{folder_path}/Self_Play_Data.h5", "r") as file:
        games_played = int(file["game_stats"][2])
    if games_played < target_games:
        warn(f"Generation ended {target_games - games_played} games short. "
             f"Re-running this generation will top it up.")


if __name__ == "__main__":
    import os
    import shutil
    import time

    # os.environ["NUMBA_CACHE_DIR"] = "numba_cache/"
    # Testing code for validation
    # from Gomoku.game import Gomoku
    # from Gomoku.config import build_config, train_config
    from TicTacToe.game import TicTacToe
    from TicTacToe.config import build_config, train_config
    from Game_Tester import Game_Tester

    folder_path = "TicTacToe/Grok_Zero_Train/0"
    # Game_Tester(TicTacToe).test()

    if os.path.exists(f"{folder_path}/Self_Play_Data.h5"):
        os.remove(f"{folder_path}/Self_Play_Data.h5")

    if os.path.exists(f"{folder_path}/Cache"):
        shutil.rmtree(f"{folder_path}/Cache")  # stale folder from the old disk cache

    with h5.File(f"{folder_path}/Self_Play_Data.h5", "w", libver="latest") as file:
        # file.create_dataset(f"max_actions", maxshape=(1,), dtype=np.uint32, data=np.zeros(1,))
        # file.create_dataset(f"num_unaugmented_games", maxshape=(1,), dtype=np.uint32, data=np.zeros(1,))
        file.create_dataset(f"game_stats", maxshape=(6,), dtype=np.uint32, data=np.zeros(6, ))

    run_self_play(TicTacToe, build_config, train_config, folder_path)

    with h5.File(f"{folder_path}/Self_Play_Data.h5", "r") as file:
        # print(file.keys())
        # print(file["game_stats"][:])
        #
        #
        # print(file['boards_0'].shape)
        # print(file['policies_0'].shape)
        # print(file['values_0'].shape)
        #
        # print(file['boards_0'][:])
        # print(file['policies_0'][:].reshape((-1, 3, 3)))
        print((len(file.keys()) - 1) // 3)
        # for i in range(file["max_actions"][0]):
        #     print(file["boards_0"][i])
        # print(len(file["boards_1"]))

        # for i in range(train_config["num_workers"]):
        #     print(file[f'boards_{i}'][1])


    def Print_Stats(folder_path):
        with h5.File(f"{folder_path}/Self_Play_Data.h5", "r", libver="latest") as file:
            max_actions, total_actions, num_unaugmented_games, player1_wins, draws, player2_wins = file["game_stats"][:]
            print("---------Game Statistics---------")
            print(f"Longest game is: {max_actions} actions long!")
            print(f"Average moves: {round(total_actions / num_unaugmented_games, 4)}")
            print(f"Player -1 winrate: {round(player1_wins / num_unaugmented_games, 4)}")
            print(f"Draw rate: {round(draws / num_unaugmented_games, 4)}")
            print(f"Player 1 winrate: {round(player2_wins / num_unaugmented_games, 4)}\n")


    Print_Stats(folder_path)
