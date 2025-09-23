import os
import time
import h5py as h5
import numpy as np

from tqdm import tqdm

import multiprocessing as mp

from MCTS import MCTS
from MCTS_Gumbel import MCTS_Gumbel

from Client_Server import Parallelized_Session, start_server, convert_to_single_info, create_shared_memory


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

        # red: might not be needed

        self.iteration_limit = self.train_config["MCTS_iteration_limit"]
        self.time_limit = self.train_config["MCTS_time_limit"]

        if train_config["use_gumbel"] is False:
            dirichlet_epsilon = 0.25
            self.mcts1: MCTS = MCTS(game=self.game,
                                    session=self.sess,
                                    use_njit=train_config["use_njit"],
                                    c_puct_init=self.train_config["c_puct_init"],
                                    use_dirichlet=True,
                                    dirichlet_alpha=self.train_config["dirichlet_alpha"],
                                    dirichlet_epsilon=dirichlet_epsilon,
                                    tau=1.0,
                                    fast_find_win=False)

            self.mcts2: MCTS = MCTS(game=self.game,
                                    session=self.sess,
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
                c_puct_init = self.mcts1.c_puct_init
                if current_move_num % 2 == 0 and current_move_num // 2 < self.train_config["num_explore_actions_first"]:
                    self.mcts1.update_hyperparams(c_puct_init=c_puct_init, tau=1.0)
                else:
                    self.mcts1.update_hyperparams(c_puct_init=c_puct_init, tau=0)

                if (current_move_num + 1) % 2 == 0 and (current_move_num + 1) // 2 < self.train_config[
                    "num_explore_actions_second"]:
                    self.mcts2.update_hyperparams(c_puct_init=c_puct_init, tau=1.0)
                else:
                    self.mcts2.update_hyperparams(c_puct_init=c_puct_init, tau=0)

                if self.game.get_next_player() == -1:
                    action, action_probs = self.mcts1.run(
                        iteration_limit=int(self.iteration_limit * (2.0 if current_move_num <= 1 else 1.0)),
                        time_limit=self.time_limit,
                        use_bar=False)
                else:
                    action, action_probs = self.mcts2.run(
                        iteration_limit=int(self.iteration_limit * (2.0 if current_move_num <= 1 else 1.0)),
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
                action = sample_actions[idx]

            self.game.do_action(action)

            winner = self.game.check_win()

            if winner == -2:  # or else there will be an error, because you are pruning a winning move
                if not self.train_config["use_gumbel"]:
                    create_new_root = self.train_config.get("create_new_root", False)
                    self.mcts1.prune_tree(action, create_new_root)
                    self.mcts2.prune_tree(action, create_new_root)
                else:
                    del self.mcts_gumbel
                    self.mcts_gumbel = self.create_MCTS_Gumbel()

            actions_count += 1
            if actions_count == self.train_config["max_actions"]:
                winner = 0 # simple cutoff in case the games get way too long

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

        target_values = (target_z + target_q) /  2 # average the q and z values ((MCTS value + win loss value (-1, 1)) / 2)

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
                   generation: int):
    np.random.seed()
    import onnxruntime as rt
    if train_config["use_inference_server"]:
        session = Parallelized_Session(worker_id,
                                       info,
                                       input_feed_info,
                                       output_feed_info, )
    else:
        providers, onnx_path = info
        session = rt.InferenceSession(onnx_path, providers=providers)

    if train_config.get("max_cache_depth", 0) not in [None, 0]: # so both 0 and None are excluded
        from Session_Cache import Cache_Wrapper
        session = Cache_Wrapper(session, folder_path + "/Cache", train_config["max_cache_depth"])
    task = Self_Play(game_class(),
                     session,
                     build_config,
                     train_config,
                     lock,
                     folder_path,
                     generation)

    task.play()

    if train_config["use_inference_server"]:
        info.close()


def convert_shape(shape):
    assert len(shape) > 0
    str_shape = ""
    for dim in shape:
        str_shape += f"{dim}x"
    return str_shape[:-1]


def run_self_play(game_class,
                  build_config,
                  train_config,
                  folder_path,
                  per_process_wait_time=1e-3):
    if not os.path.exists(f"{folder_path}/Self_Play_Data.h5"):
        raise ValueError("Dataset file hasn't been created. Self play depends on that file!")

    with h5.File(f"{folder_path}/Self_Play_Data.h5") as dataset_file:
        num_games_left = int(train_config["games_per_generation"] - dataset_file["game_stats"][2])

    if num_games_left <= 0:
        print(f"Finished generating {train_config['games_per_generation']} games!")
        return

    generation = int(folder_path.split("/")[-1])
    num_workers = train_config["num_workers"]
    if num_workers > mp.cpu_count() // 2 and mp.get_start_method() == "spawn":
        num_workers = mp.cpu_count() // 2

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
            if train_config.get("mixed_precision") == "mixed_float16":
                providers[0][1]["trt_fp16_enable"] = True
        elif "CUDAExecutionProvider" in available_providers:
            providers = [
                'CUDAExecutionProvider',
                'CPUExecutionProvider']
        elif "DmlExecutionProvider" in available_providers:
            providers = [
                'DmlExecutionProvider',
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
    if train_config["use_inference_server"]:
        shms = create_shared_memory(batched_input_feed_info, batched_output_feed_info, num_workers)
        server = mp.Process(target=start_server, args=(batched_input_feed_info,
                                                       batched_output_feed_info,
                                                       shms,
                                                       providers,
                                                       onnx_file_path,
                                                       per_process_wait_time))
        server.start()

    lock = mp.Lock()
    jobs = []

    for _ in range(num_games_left):
        if len(jobs) < num_workers:
            worker_id = len(jobs)
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
                                      generation),
                                name=f"{worker_id}")
            worker.start()
            jobs.append(worker)

    while num_games_left > 0:
        if len(jobs) == num_workers:
            alive_jobs = []
            dead_worker_ids = []
            for worker in jobs:
                if worker.is_alive():
                    alive_jobs.append(worker)
                else:
                    dead_worker_ids.append(int(worker.name))
                    worker.join()
                    if worker.exitcode != 0:
                        server.terminate()
                        for shm in shms:
                            shm.unlink()
                        raise RuntimeError("Main process stopped as self play worker ran into an error")
                    num_games_left -= 1
                    bar.update(1)

            if len(alive_jobs) != len(jobs) and num_games_left - len(alive_jobs) > 0:
                for dead_worker_id in dead_worker_ids:
                    new_worker = mp.Process(target=self_play_task, args=(dead_worker_id,
                                                                         shms[dead_worker_id] if train_config[
                                                                             "use_inference_server"] else [providers,
                                                                                                           onnx_file_path],
                                                                         game_class,
                                                                         convert_to_single_info(batched_input_feed_info),
                                                                         convert_to_single_info(batched_output_feed_info),
                                                                         build_config,
                                                                         train_config,
                                                                         lock,
                                                                         folder_path,
                                                                         generation),
                                            name=f"{dead_worker_id}")
                    new_worker.start()
                    alive_jobs.append(new_worker)
                jobs = alive_jobs

    for worker in jobs:
        worker.join()
        if worker.exitcode != 0:
            server.terminate()
            for shm in shms:
                shm.unlink()
            raise RuntimeError("Main process stopped as self play worker ran into an error")

    if train_config["use_inference_server"]:
        server.terminate()
        for shm in shms:
            shm.unlink()


if __name__ == "__main__":
    from diskcache import Cache
    import os
    import shutil
    import time

    # os.environ["NUMBA_CACHE_DIR"] = "numba_cache/"
    # Testing code for validation
    # from Gomoku.Gomoku import Gomoku, build_config, train_config
    from TicTacToe.Tictactoe import TicTacToe, build_config, train_config
    from Game_Tester import Game_Tester

    folder_path = "TicTacToe/Grok_Zero_Train/0"
    cache = Cache(folder_path + "/Cache")
    cache.close()
    # Game_Tester(TicTacToe).test()

    if os.path.exists(f"{folder_path}/Self_Play_Data.h5"):
        os.remove(f"{folder_path}/Self_Play_Data.h5")

    if os.path.exists(f"{folder_path}/Cache"):
        shutil.rmtree(f"{folder_path}/Cache")

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
