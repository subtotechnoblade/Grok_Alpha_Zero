import os
import h5py as h5
import numpy as np
from tqdm import tqdm
from numba import njit
import onnxruntime as rt
import multiprocessing as mp

from MCTS import MCTS

from Gomoku import Gomoku

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
        self.mcts: MCTS = MCTS(self.game,
                               self.build_config,
                               self.sess,
                               c_puct_init=self.train_config["c_puct_init"],
                               use_dirichlet=True,
                               dirichlet_alpha=self.train_config["dirichlet_alpha"],
                               tau=1.0,
                               fast_find_win=False,
                               use_njit=self.train_config["use_njit"])

    def play(self):
        board_states = []
        improved_policies = []
        target_values = []

        winner = -2
        while winner == -2:
            board_states.append(self.game.get_input_state().copy())

            current_move_num = len(self.game.action_history)
            if current_move_num < self.train_config["num_explore_moves"]:
                tau = 1.0 - (0.5 * (current_move_num / self.train_config["num_explore_moves"]))
                self.mcts.update_hyperparams(self.mcts.c_puct_init, tau)
            else:
                self.mcts.update_hyperparams(self.mcts.c_puct_init, 5e-3)

            action, move_probs = self.mcts.run(iteration_limit=self.iteration_limit,
                                               time_limit=self.time_limit,
                                               use_bar=False)

            move_probs = map(lambda x: x[:2], move_probs) # This takes the first and seconds element of which is the [action, prob]
            improved_policy = self.game.compute_policy_improvement(move_probs)
            improved_policies.append(improved_policy)

            target_values.append(self.game.get_current_player()) # Important that this is before do_action()
            # We can safely say that target_values are the players that played the move, not the next player

            self.game.do_action(action)
            print(self.game.board)

            winner = self.game.check_win()

            if winner == -2:
                self.mcts.prune_tree(action) # or else there will be an error because you are pruning a winning move
                # there are no more moves after a winning move

            if winner != -2: # there is a winner
                print(f"Player: {winner} won")
                board_states = np.array(board_states, dtype=board_states[0].dtype)
                improved_policies = np.array(improved_policies)
                target_values = np.array(target_values, dtype=np.int8)

                if winner ==  target_values[-1] == -1: # if player -1 just won
                    target_values *= -1 # Flip it so that the player that won, evaluates to 1 (winner)
                # else the player that played was 1, and won which is 1, thus no need to invert

        # Assume that a .h5 file has been created and the max moves dataset is already created

        with self.lock, h5.File(f"{self.folder_path}/{self.generation}/Self_Play_Data.h5", "r+") as file:
            game_length = len(self.game.action_history)
            if file["max_moves"][0] < game_length:
                file["max_moves"][0] = game_length

            dataset_name = f"{(len(file.keys()) - 1) // 3}" # starts from 0

            file.create_dataset(f"board_states_{dataset_name}",
                                shape=(game_length, *self.game.get_input_state().shape),
                                dtype=board_states.dtype,
                                data=board_states)

            file.create_dataset(f"policies_{dataset_name}",
                                shape=(game_length, self.game.policy_shape[0]),
                                dtype=np.float32,
                                data=improved_policies)

            file.create_dataset(f"values_{dataset_name}",
                                shape=(game_length, 1),
                                dtype=np.float32,
                                data=target_values)


def self_play_task(worker_id,
                   shm,
                   game_class,
                   input_feed_info: dict,
                   output_feed_info: dict,
                   build_config:dict,
                   train_config:dict,
                   lock: mp.Lock,
                   folder_path:str,
                   generation:int):


    parallelized_session = Parallelized_Session(worker_id,
                                                shm,
                                                input_feed_info,
                                                output_feed_info,)

    # print(game_id)
    task = Self_Play(game_class(),
                     parallelized_session,
                     build_config,
                     train_config,
                     lock,
                     folder_path,
                     generation)
    task.play()

def run_self_play(game_class,
                  build_config,
                  train_config,
                  folder_path,
                  generation,
                  num_processes=1):
    if not os.path.exists(f"{folder_path}/{generation}/Self_Play_Data.h5"):
        raise ValueError("Dataset file hasn't been created. Self play depends on that file")

    with h5.File(f"{folder_path}/{generation}/Self_Play_Data.h5") as file:
        num_games_left = train_config["games_per_generation"] - ((len(file.keys()) - 1) // 3)

    bar = tqdm(total=num_games_left)
    if num_games_left < num_processes:
        num_processes = num_games_left

    embed_size, num_heads, num_layers = build_config["embed_size"], build_config["num_heads"], build_config[
        "num_layers"]
    onnx_file_path = f"{folder_path}/{generation}/model.onnx"
    if train_config["use_gpu"]:
        if train_config["use_tensorrt"]:
            max_shape = train_config["num_workers"]
            onnx_file_path = f"{folder_path}/{generation}/TRT_cache/model_ctx.onnx"
            providers = [
                ('TensorrtExecutionProvider', {
                    "trt_engine_cache_enable": True,
                    "trt_dump_ep_context_model": True,
                    "trt_builder_optimization_level": 5,
                    "trt_auxiliary_streams": 0,

                    "trt_ep_context_file_path": f"{folder_path}/{generation}/TRT_cache/",
                    "trt_profile_min_shapes": f"inputs:1x15x15,input_state:{num_layers}x2x1x{embed_size},input_state_matrix:{num_layers}x1x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",
                    "trt_profile_max_shapes": f"inputs:{max_shape}x15x15,input_state:{num_layers}x2x{max_shape}x{embed_size},input_state_matrix:{num_layers}x{max_shape}x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",
                    "trt_profile_opt_shapes": f"inputs:{max_shape}x15x15,input_state:{num_layers}x2x{max_shape}x{embed_size},input_state_matrix:{num_layers}x{max_shape}x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",
                }),
                'CUDAExecutionProvider',
                'CPUExecutionProvider']
        else:
            providers = [
                'CUDAExecutionProvider',
                'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    batched_input_feed_info = {"inputs": [[-1, 15, 15], np.float32],
                               "input_state": [[num_layers, 2, -1, embed_size], np.float32],
                               "input_state_matrix":[[num_layers, -1, num_heads, embed_size // num_heads, embed_size // num_heads], np.float32]}
    batched_output_feed_info = {"policy": [-1, 225],
                                "value": [-1, 1],
                                "output_state": [num_layers, 2, -1, embed_size],
                                "output_state_matrix": [num_layers, -1, num_heads, embed_size // num_heads, embed_size // num_heads]
                                }
    print(train_config["num_workers"])
    shms = create_shared_memory(batched_input_feed_info, batched_output_feed_info, train_config["num_workers"])
    lock = mp.Lock()
    jobs = []
    for worker_id in range(num_processes):
        p = mp.Process(target=self_play_task, args=(worker_id,
                                                    shms[worker_id],
                                                    game_class,
                                                    convert_to_single_info(batched_input_feed_info),
                                                    convert_to_single_info(batched_output_feed_info),
                                                    build_config,
                                                    train_config,
                                                    lock,
                                                    folder_path,
                                                    generation))
        p.start()
        jobs.append(p)
    num_games_left -= num_processes

    server = mp.Process(target=start_server, args=(batched_input_feed_info,
                                                   batched_output_feed_info,
                                                   shms,
                                                   providers,
                                                   onnx_file_path))
    server.start()

    for worker_id, p in enumerate(jobs):
        if not p.is_alive():
            bar.update(1)
        p.join()


    server.terminate()

    with h5.File("Gomoku/Grok_Zero_Train/0/Self_Play_Data.h5", "r") as file:
        print(file.keys())
        print(file["max_moves"][0])
        # print(file['board_states_0'])
        print(len(file['board_states_0']))
        print(len(file['policies_0']))
        print(len(file['values_0']))

    # raise ValueError
    # while num_games_left > 0:
    #     for process in jobs:
    #         if not process.is_alive():
    #
    #             p = mp.Process(target=self_play_task,
    #                            args=(game_class, build_config, train_config, lock, folder_path, generation))
    #             p.start()
    #             jobs.append(p)
    #             num_games_left -= 1
    #             p.join()







    # results = [pool.apply_async(self_play_task, args=(game_id, game_class, build_config, train_config, lock, folder_path, generation)) for game_id in range(train_config["games_per_generation"])]
    # for r in results:
    #     r.wait()

if __name__== "__main__":
    import time
    # Testing code for validation
    from Gomoku.Gomoku import Gomoku, build_config, train_config
    if os.path.exists("Gomoku/Grok_Zero_Train/0/Self_Play_Data.h5"):
        os.remove("Gomoku/Grok_Zero_Train/0/Self_Play_Data.h5")

    with h5.File("Gomoku/Grok_Zero_Train/0/Self_Play_Data.h5", "w", libver="latest") as file:
        file.create_dataset(f"max_moves", maxshape=(1,), dtype=np.int32, data=np.zeros(1,))

    folder_path = "Gomoku/Grok_Zero_Train"
    run_self_play(Gomoku, build_config, train_config, folder_path, 0)











