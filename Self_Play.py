import os
import h5py as h5
import numpy as np

from tqdm import tqdm

import onnxruntime as rt
import multiprocessing as mp

from MCTS import MCTS

from Gomoku import Gomoku

from Session_Cache import Cache_Wrapper
from Client_Server import Parallelized_Session, start_server, convert_to_single_info, create_shared_memory


class Self_Play:
    def __init__(self,
                 game,
                 sess,
                 build_config:dict,
                 train_config:dict,
                 lock: mp.Lock,
                 folder_path:str,
                 generation:int):
        self.game: Gomoku = game
        self.sess = sess
        self.build_config = build_config
        self.train_config = train_config
        self.lock = lock
        self.folder_path = folder_path
        self.generation = generation

        # red: might not be needed

        self.iteration_limit = self.train_config["MCTS_iteration_limit"]
        self.time_limit = self.train_config["MCTS_time_limit"]

        embed_size, num_heads, num_layers = build_config["embed_size"], build_config["num_heads"], build_config[
            "num_layers"]
        RNN_state1 = [np.zeros((num_layers, 2, 1, embed_size), dtype=np.float32),
                                            np.zeros((num_layers, 1, num_heads, embed_size // num_heads, embed_size // num_heads), dtype=np.float32)]
        self.mcts1: MCTS = MCTS(self.game,
                               RNN_state1,
                               self.sess,
                               c_puct_init=self.train_config["c_puct_init"],
                               use_dirichlet=True,
                               dirichlet_alpha=self.train_config["dirichlet_alpha"],
                               tau=1.0,
                               fast_find_win=False)
        # RNN_state2 = [np.zeros((num_layers, 2, 1, embed_size), dtype=np.float32),
        #                                     np.zeros((num_layers, 1, num_heads, embed_size // num_heads, embed_size // num_heads), dtype=np.float32)]
        # self.mcts2: MCTS = MCTS(self.game,
        #                        RNN_state2,
        #                        self.sess,
        #                        c_puct_init=self.train_config["c_puct_init"],
        #                        use_dirichlet=True,
        #                        dirichlet_alpha=self.train_config["dirichlet_alpha"],
        #                        tau=1.0,
        #                        fast_find_win=False)

    def play(self):
        board_states = []
        improved_policies = []
        target_values = []

        actions_count = 0
        winner = -2
        while winner == -2 and actions_count < self.train_config["max_actions"]:
            board_states.append(self.game.get_input_state().copy())

            current_move_num = len(self.game.action_history)

            if current_move_num % 2 == 0 and current_move_num // 2 < self.train_config["num_explore_actions_first"]:
                tau = 1.0 - (0.75 * ((current_move_num // 2) / self.train_config["num_explore_actions_first"]))

                self.mcts1.update_hyperparams(self.mcts1.c_puct_init, tau)
            else:
                self.mcts1.update_hyperparams(self.mcts1.c_puct_init, 0)

            if (current_move_num + 1) % 2 == 0 and (current_move_num + 1) // 2 < self.train_config["num_explore_actions_second"]:
                tau = 1.0 - (0.75 * (((current_move_num + 1) // 2) / self.train_config["num_explore_actions_second"]))
                self.mcts1.update_hyperparams(self.mcts1.c_puct_init, tau)
            else:
                self.mcts1.update_hyperparams(self.mcts1.c_puct_init, 0)

            # if self.game.get_current_player() == -1:
            action, move_probs = self.mcts1.run(iteration_limit=self.iteration_limit,
                                               time_limit=self.time_limit,
                                               use_bar=True)
            # else:
            #     action, move_probs = self.mcts2.run(iteration_limit=self.iteration_limit,
            #                                        time_limit=self.time_limit,
            #                                        use_bar=False)

            move_probs = map(lambda x: x[:2], move_probs) # This takes the first and seconds element of which is the [action, prob]
            improved_policy = self.game.compute_policy_improvement(move_probs)
            improved_policies.append(improved_policy)

            target_values.append(self.game.get_current_player()) # Important that this is before do_action()
            # We can safely say that target_values are the players that played the move, not the next player
            if current_move_num == 0 and self.train_config.get("opening_actions"):
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

            if winner == -2:
                self.mcts1.prune_tree(action) # or else there will be an error because you are pruning a winning move
                # self.mcts2.prune_tree(action) # or else there will be an error because you are pruning a winning move

                # there are no more moves after a winning move
            # else:
            # print(f"Player: {winner} won")
            actions_count += 1
            if actions_count == self.train_config["max_actions"]:
                winner = 0

        # there is a winner
        board_states = np.array(board_states, dtype=board_states[0].dtype)
        improved_policies = np.array(improved_policies, dtype=np.float32)

        target_values = np.array(target_values, dtype=np.float32).reshape((-1, 1))
        if winner == target_values[-1][0] == -1: # if player -1 just won
            target_values *= -1.0 # Flip it so that the player that won, evaluates to 1 (winner)
        elif winner == 0: # if it a draw or
            target_values[:] = 0.0
        # else the player that played was 1, and won which is 1, thus no need to invert
        # print(winner)
        # print(board_states)
        # print(target_values)
        # print(improved_policies)
        # raise ValueError
        # augmentation
        augmented_board_states, augmented_policies = self.game.augment_sample(board_states, improved_policies)
        augmented_values = np.repeat(np.expand_dims(target_values, 0), repeats=augmented_policies.shape[0], axis=0)

        if augmented_board_states.shape[:2] != augmented_values.shape[:2]:
            print(f"The 0th and 1st dim should the same got: {augmented_board_states.shape[2:]}, {augmented_values.shape[2:]}")

        # Assume that a .h5 file has been created and the max moves dataset is already created
        with self.lock, h5.File(f"{self.folder_path}/Self_Play_Data.h5", "r+") as file:
            game_length = len(self.game.action_history)

            # [max_actions, total_actions, num_unaugmented_games, player -1 wins, draws, player 1 wins]
            if file["game_stats"][0] < game_length:
                file["game_stats"][0] = game_length

            file["game_stats"][1] += board_states.shape[0] # adds to
            file["game_stats"][2] += 1 # adds a game to num_unaugmented_games

            file["game_stats"][winner + 4] += 1 # adding winners/ draws

            dataset_name = (len(file.keys()) - 1) // 3 # starts from 0

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
                   build_config:dict,
                   train_config:dict,
                   lock: mp.Lock,
                   folder_path:str,
                   generation:int):
    np.random.seed()

    if train_config["use_inference_server"]:
        session = Parallelized_Session(worker_id,
                                       info,
                                       input_feed_info,
                                       output_feed_info,)
    else:
        providers, onnx_path = info
        session = rt.InferenceSession(onnx_path, providers=providers)
    session = Cache_Wrapper(session, folder_path + "/Cache")
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
                  folder_path):
    if not os.path.exists(f"{folder_path}/Self_Play_Data.h5"):
        raise ValueError("Dataset file hasn't been created. Self play depends on that file!")

    with h5.File(f"{folder_path}/Self_Play_Data.h5") as dataset_file:
        num_games_left = train_config["games_per_generation"] - dataset_file["game_stats"][2]

    if num_games_left <= 0:
        print(f"Finished generating {train_config['games_per_generation']} games!")
        return

    generation = int(folder_path.split("/")[-1])
    num_workers = train_config["num_workers"]

    bar = tqdm(total=num_games_left, desc="Generating self play games")
    if num_games_left < num_workers:
        num_workers = num_games_left

    game = game_class()
    board_shape = game.board.shape
    policy_shape = game.policy_shape
    str_board_shape = convert_shape(board_shape)
    del game


    embed_size, num_heads, num_layers = build_config["embed_size"], build_config["num_heads"], build_config[
        "num_layers"]
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
                    "trt_profile_min_shapes": f"inputs:1x{str_board_shape},input_state:{num_layers}x2x1x{embed_size},input_state_matrix:{num_layers}x1x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",
                    "trt_profile_max_shapes": f"inputs:{max_shape}x{str_board_shape},input_state:{num_layers}x2x{max_shape}x{embed_size},input_state_matrix:{num_layers}x{max_shape}x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",
                    "trt_profile_opt_shapes": f"inputs:{max_shape}x{str_board_shape},input_state:{num_layers}x2x{max_shape}x{embed_size},input_state_matrix:{num_layers}x{max_shape}x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",
                }),
                'CUDAExecutionProvider',
                'CPUExecutionProvider']
        else:
            providers = [
                'CUDAExecutionProvider',
                'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    batched_input_feed_info = {"inputs": [[-1, *board_shape], np.float32],
                               "input_state": [[num_layers, 2, -1, embed_size], np.float32],
                               "input_state_matrix":[[num_layers, -1, num_heads, embed_size // num_heads, embed_size // num_heads], np.float32]}
    batched_output_feed_info = {"policy": [-1, *policy_shape],
                                "value": [-1, 1],
                                "output_state": [num_layers, 2, -1, embed_size],
                                "output_state_matrix": [num_layers, -1, num_heads, embed_size // num_heads, embed_size // num_heads]
                                }
    print(f"Running with {num_workers} workers for {num_games_left} games with {onnx_file_path} for generation: {generation}!\n")

    sess_options = rt.SessionOptions()
    if not train_config["use_gpu"]: # we are using the CPU for self play
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1

    shms = []
    if train_config["use_inference_server"]:
        shms = create_shared_memory(batched_input_feed_info, batched_output_feed_info, num_workers)
        server = mp.Process(target=start_server, args=(batched_input_feed_info,
                                                       batched_output_feed_info,
                                                       shms,
                                                       providers,
                                                       sess_options,
                                                       onnx_file_path,
                                                       0.01))
        server.start()

    lock = mp.Lock()
    jobs = []

    for _ in range(num_games_left):
        if len(jobs) < num_workers:
            worker_id = len(jobs)
            worker = mp.Process(target=self_play_task,
                                args=(worker_id,
                                      shms[worker_id] if train_config["use_inference_server"] else [providers, onnx_file_path],
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
                    num_games_left -= 1
                    bar.update(1)


            if len(alive_jobs) != len(jobs) and num_games_left - len(alive_jobs) > 0:
                for dead_worker_id in dead_worker_ids:
                    new_worker = mp.Process(target=self_play_task, args=(dead_worker_id,
                                                                         shms[dead_worker_id] if train_config["use_inference_server"] else [providers, onnx_file_path],
                                                                         game_class,
                                                                         convert_to_single_info(batched_input_feed_info),
                                                                         convert_to_single_info(batched_output_feed_info),
                                                                         build_config,
                                                                         train_config,
                                                                         lock,
                                                                         folder_path,
                                                                         generation),
                                            name=f"{dead_worker_id}")

                    alive_jobs.append(new_worker)
                    new_worker.start()
                jobs = alive_jobs

    for worker in jobs:
        worker.join()

    if train_config["use_inference_server"]:
        server.terminate()
        for shm in shms:
            shm.unlink()

if __name__== "__main__":
    from diskcache import Cache
    import os
    import shutil
    import time
    # Testing code for validation
    from Gomoku.Gomoku import Gomoku, build_config, train_config
    # from TicTacToe.Tictactoe import TicTacToe, build_config, train_config
    from Game_Tester import Game_Tester

    folder_path = "Gomoku/Grok_Zero_Train/1"
    cache = Cache(folder_path + "/Cache")
    cache.close()
    Game_Tester(Gomoku).test()


    if os.path.exists(f"{folder_path}/Self_Play_Data.h5"):
        os.remove(f"{folder_path}/Self_Play_Data.h5")

    # if os.path.exists(f"{folder_path}/Cache"):
    #     shutil.rmtree(f"{folder_path}/Cache")

    with h5.File(f"{folder_path}/Self_Play_Data.h5", "w", libver="latest") as file:
        # file.create_dataset(f"max_actions", maxshape=(1,), dtype=np.uint32, data=np.zeros(1,))
        # file.create_dataset(f"num_unaugmented_games", maxshape=(1,), dtype=np.uint32, data=np.zeros(1,))
        file.create_dataset(f"game_stats", maxshape=(6,), dtype=np.uint32, data=np.zeros(6,))

    run_self_play(Gomoku, build_config, train_config, folder_path)

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










