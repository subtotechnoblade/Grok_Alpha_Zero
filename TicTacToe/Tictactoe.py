import numpy as np
from numba import njit
import numba as nb
from numba import types
import time

# make zeros array np.zeros(shape)
# make ones array np.ones(shape)
# make array from list np.array(list)

# human friendly board, not computer friendly
# [["E", "E", "E"]]
# [["E", "X", "O"]]
# [["E", "X", "O"]]
#
# [[0, 0, 0],
#  [0, 0, 0],
#  [0, 0, 0]]hi
# pieces with -1, 0, 1

# board = np.zeros((3, 3), dtype=np.int8)
# numpy array board
# [[0 0 0]
#  [0 0 0]
#  [0 0 0]]
# print("Starting Board")
# print(board)
# print("--"*10)
# Next thing
# #
# board[0][0] = -1
# board[1][1] = 1
# board[0][1] = -1
# board[0][2] = 1
# board[2][0] = -1
# board[1][0] = 1
# board[1][2] = -1
# board[2][2] = 1
# board[2][1] = -1
# print(board)
# print("draw")

# the board
# coordinate x, y
# and the current player as -1 or 1
# places the move onto the board
# returns the board with the newly placed board
#
# def b_func(board, x, y, player):
#     board = [[0, 0, 1],
#              [0, 1, 0],
#              [1, 0, 0]]
#     board = [[1, 0, 0],
#              [0, 1, 0],
#              [0, 0, 1]]
#     board[x][y] = player
#     print(board)
#
# b_func(2, 1, 1)

# outerboard = np.zeros((3, 3), dtype=np.int8)
# def func(board, x, y, player):
#     board[y][x] = player
#     return board

# function vs method
# function

# def foo(inputs):
#     return inputs ** 2

# class
# class House:
#     def __init__(self, new_owner): # constructor
#         self.owner = "Jessica" # Jessica
#         self.owner = new_owner # Brian overwrites Jessica
#
# house = House("Brian") # this is calling constructor
# print(f"Ha Ha {house.owner} is the new owner")

build_config = {"num_resnet_layers": 1,

                "use_stablemax": True,
                "use_grok_fast": True,
                "use_orthograd": True,
                "grok_lambda": 4.5,  # This is for grok fast, won't be used if model is Grok_Fast_EMA_Model
                }

train_config = {
    "total_generations": 6,  # Total number of generations, the training can be stopped and resume at any moment
    # a generation is defined by a round of self play, padding the dataset, model training, converting to onnx

    # Self Play variables
    "games_per_generation": 1000,  # number of self play games until we re train the network
    "max_actions": 9,  # Note that this should be
    "num_explore_actions_first": 2,
    # This is for tictactoe, a good rule of thumb is 10% to 20% of the average length of a game
    "num_explore_actions_second": 1,
    # for a random player player -1 almost always wins, so player 1 should try playing the best move

    "use_gpu": False,  # Change this to false to use CPU for self play and inference
    "use_tensorrt": False,  # Assuming use_gpu is True, uses TensorrtExecutionProvider
    # change this to False to use CUDAExecutionProvider
    "use_inference_server": True,
    # if an extremely large model is used, because of memory constraints, set this to True
    "max_cache_depth": 0,  # maximum depth in the search of the neural networks outputs we should cache
    "num_workers": 12,  # Number of multiprocessing workers used to self play

    # MCTS variables
    "MCTS_iteration_limit": 27,  # The number of iterations MCTS runs for. Should be 2 to 10x the number of starting legal moves
    # True defaults to iteration_limit = 3 * len(starting legal actions)
    "MCTS_time_limit": None,  # Not recommended to use for training, True defaults to 30 seconds
    "use_njit": None,  # None will automatically infer what is supposed to be use for windows/linux

    "use_gumbel": False,  # use gumbel according to https://openreview.net/pdf?id=bERaNdoegnO, time_limit won't be used
    # These params will only be used when use_gumbel is set to True
    "m": 9,  # Number of actions sampled in the first stage of sequential halving
    "c_visit": 50.0,
    "c_scale": 1.0,

    # These params will be used when use_gumbel is set to False
    "c_puct_init": 1.25,  # (shouldn't change) Exploration constant lower -> exploitation, higher -> exploration
    "dirichlet_alpha": 0.8,  # should be around (10 / average moves per game)

    # "opening_actions": [[[1, 1], 0.4]],  # starting first move in the format [[action1, prob0], [action1, prob1], ...],
    # if prob doesn't add up to 1, then the remaining prob is for the MCTS move

    "num_previous_generations": 2,  # The previous generation's data that will be used in training
    "train_percent": 1.0,  # The percent used for training after the test set is taken
    "train_decay": 0.9,
    # The decay rate for previous generations of data previous_train_percent = current_train_percent * train_decay
    "test_percent": 0.1,  # The percent of a dataset that will be used for validation
    "test_decay": 0.9,
    # The decay rate for previous generations of data previous_test_percent = current_test_percent * test_decay

    "mixed_precision": None,  # None for no mixed precision, mixed_float16 for float16
    "train_batch_size": 512,  # The number of samples in a batch for training in parallel
    "test_batch_size": None,  # If none, then train_batch_size will be used for the test batch size
    "gradient_accumulation_steps": None,
    "learning_rate": 7e-4,  # Depending on how many layers you use. Recommended to be between 1e-3 to 5e-4
    "decay_lr_after": 4,  # When the n generations pass,... learning rate will be decreased by lr decay
    "lr_decay": 0.75,  # multiplies this to learning rate every decay_lr_after
    "beta_1": 0.9,  # DO NOT TOUCH unless you know what you are doing
    "beta_2": 0.99,  # DO NOT TOUCH. This determines whether it groks or not. Hovers between 0.98 to 0.995
    "optimizer": "Nadam",  # optimizer options are ["Adam", "AdamW", "Nadam"]
    "train_epochs": 15,  # The number of epochs for training
}


class TicTacToe:
    def __init__(self):
        self.next_player = -1
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.action_history = []
        # doesn't return anything

        self.policy_shape = (9,)

    def get_next_player(self):
        return self.next_player

    def input_action(self):
        while True:
            try:
                coords = input("Move:").split(" ")
                coords[0], coords[1] = int(coords[0]), int(coords[1])
                x, y = coords
                if (0 > x > 2 or 0 < y > 2) or self.board[coords[1]][coords[0]] != 0:
                    print("Illegal move given")
                    continue
                return coords
            except:
                print("Invalid Move")
                continue

    def get_legal_actions(self):  # -> list[tuple[int, int], ...] or np.array:
        """
        self.total_action is a set
        :return: a list of actions [action0, action1]
        """
        return self.get_legal_actions_MCTS(self.board, -self.next_player, np.array(self.action_history))

    @staticmethod
    @njit(cache=True)
    def get_legal_actions_MCTS(board, current_player, action_history):
        return np.argwhere(board == 0)[:, ::-1]

    @staticmethod
    @njit(cache=True)
    def get_legal_actions_policy_MCTS(board: np.array,
                                      current_player: int,
                                      action_history: np.array,
                                      policy: np.array,
                                      normalize=True,
                                      shuffle: bool = False):

        legal_actions = np.argwhere(board == 0)[:, ::-1]
        legal_policy = policy[board.reshape(-1) == 0]

        if normalize:
            legal_policy /= np.sum(legal_policy)

        if shuffle:
            random_indexes = np.random.permutation(len(legal_actions))
            legal_actions = legal_actions[random_indexes]
            legal_policy = legal_policy[random_indexes]

        return legal_actions, legal_policy

    def do_action(self, action):  # must conform to this format - Brian
        x, y = action
        if self.board[y][x] != 0:
            raise ValueError("Illegal move")
        self.board[y][x] = self.next_player
        self.next_player = self.next_player * -1
        self.action_history.append(action)

    @staticmethod
    @njit(cache=True)
    def do_action_MCTS(board: np.array, action: np.array, next_player: int) -> np.array:
        x, y = action
        board[y][x] = next_player
        return board

    def get_input_state(self):
        return self.get_input_state_MCTS(self.board, -self.next_player, np.array(self.action_history))

    @staticmethod
    @njit(cache=True)
    def get_input_state_MCTS(board: np.array, current_player: int, action_history: np.array):
        # Used for retrieving the state for any child nodes (not the root)
        # just return the board from the inputs
        current_player_board = np.expand_dims(np.ones_like(board, dtype=board.dtype) * -current_player, -1)
        return np.concatenate((current_player_board, np.expand_dims(board, -1)), axis=-1)

    def _check_row(self, row):  # A function for checking if there's a win on each row.
        return row[0] != 0 and row[0] == row[1] == row[2]


    def check_win(self):
        # Note that self.current player is actually the next player
        # -self.current_player because we've just played a move which changes to the next player
        # thus if you found a win you have to return the previous player since that player just played -Brian
        current_player = -self.next_player
        for row in self.board:
            if self._check_row(row):  # Pulling out func from LALA-land
                "                    ^ Jessy, just letting you know that 'is True' is not necessary for python"
                "It's equally valid to do 'if self._check_row(row):' - Brian"
                return current_player  # Someone has won! :D

        for column_index in range(3):  # Checking if someone has won in columns
            column = self.board[:, column_index]
            if self._check_row(column):  # Pulling out func from LALA-land pt2
                return current_player  # Someone has won, yippee :3

        diag1 = np.diag(self.board)  # Checking if someone has won in diagonals
        if self._check_row(diag1):  # Pulling out func from dark magic
            return current_player  # Someone has won :P

        diag2 = np.diag(np.fliplr(self.board))
        if self._check_row(diag2):  # Pulling func from Narnia
            return current_player  # Someone has won :)

        flattened_board = self.board.reshape((9,))  # Reshape board into 9 items, this is to check for draw
        for value in flattened_board:
            if value == 0:  # Still empty spaces on the board, stop checking!!
                break
        else:
            return 0  # return a draw if break never happens - Brian
        return -2  # Game continues :3

    @staticmethod
    @njit(cache=True)
    def check_win_MCTS(board, current_player, action_history):
        for row in board:
            if row[0] != 0 and row[0] == row[1] == row[2]:
                return current_player

        for column_index in range(3):
            column = board[:, column_index]
            if column[0] != 0 and column[0] == column[1] == column[2]:
                return current_player

        diag1 = np.diag(board)
        if diag1[0] != 0 and diag1[0] == diag1[1] == diag1[2]:
            return current_player

        diag2 = np.diag(np.fliplr(board))
        if diag2[0] != 0 and diag2[0] == diag2[1] == diag2[2]:
            return current_player

        flattened_board = board.reshape((9,))
        for value in flattened_board:
            if value == 0:
                break
        else:
            return 0

        return -2

    def compute_policy_improvement(self, statistics):
        # given statistic=[[action, probability], ...] compute the new policy which should be of shape=self.policy_shape
        # example for tic tac toe statistics=[[[0, 0], 0.1], [[1, 0], 0.2], ...]
        # return [0.1, 0.2, ...]
        # this should map the action and probability to a probability distribution
        new_policy = np.zeros((3, 3), np.float32)
        # new_policy1 = np.zeros(self.policy_shape, np.float32)
        for (x, y), prob in statistics:
            # new_policy1[x + 3 * y] = prob
            new_policy[y][x] = prob

        # if not np.array_equal(new_policy1, new_policy.flatten()):
        #     print(new_policy1)
        #     print(new_policy.flatten())
        #     raise ValueError("Policy is incorrect")
        return new_policy.flatten()

    @staticmethod
    @njit(cache=True)
    def augment_sample_fn(states: np.array, policies: np.array):
        policies = policies.reshape((-1, 3, 3))  # we need
        # to reshape this because we can only rotate a matrix, not a vector

        augmented_boards = []
        augmented_policies = []

        for action_id in range(states.shape[0]):
            state = states[action_id]
            # augmented_board = np.zeros((8, board_shape))
            augmented_state = [state, np.flipud(state), np.fliplr(state)]

            policy = policies[action_id]
            augmented_policy = [policy, np.flipud(policy), np.fliplr(policy)]

            for k in range(1, 4):
                rot_state = np.rot90(state, k)
                augmented_state.append(rot_state)

                rot_policy = np.rot90(policy, k)
                augmented_policy.append(rot_policy)

                if k == 1:
                    augmented_state.append(np.flipud(rot_state))
                    augmented_state.append(np.fliplr(rot_state))

                    augmented_policy.append(np.flipud(rot_policy))
                    augmented_policy.append(np.fliplr(rot_policy))
            augmented_boards.append(augmented_state)

            augmented_policies.append(augmented_policy)
        return augmented_boards, augmented_policies

    def augment_sample(self, input_states, policies):
        # Note that values don't have to be augmented since they are the same regardless of how a board is rotated
        augmented_boards, augmented_policies = self.augment_sample_fn(input_states, policies)
        return np.array(augmented_boards, dtype=self.board.dtype).transpose([1, 0, 2, 3, 4]), np.array(
            augmented_policies, dtype=np.float32).reshape((-1, 8, 9)).transpose([1, 0, 2])
        # return np.expand_dims(input_states, 0).astype(self.board.dtype), np.expand_dims(policies, 0).astype(np.float32)

if __name__ == "__main__":
    # test your code here
    # from Game_Tester import Game_Tester
    # tester = Game_Tester(TicTacToe)
    # tester.test()

    game = TicTacToe()
    game.board = np.array([[1, -1,  1],
     [-1, -1, -1],
     [0,  1,  0]])
    game.next_player = 1
    print(game.check_win())
    print(game.check_win_MCTS(game.board, -game.next_player, None))
    # print(game.get_terminal_actions)
    # # game.make_terminal_actions_MCTS()
    # action_histories = np.expand_dims(game.get_legal_actions(), 0)
    # print(game.get_terminal_actions(action_histories, game.board, game.next_player))
    # print(game.get_input_state())
    # print(game.board)
    # board = np.zeros((2, 2))
    # board = np.array([0.0, 0.0], dtype=np.uint8)
    # print(board.dtype)

    # dtype -> data type
    # integer
    # np.uint8 0 to 255
    # np.int16 short, int
    # np.int32 long
    # np.int64 long long

    #

    # game.do_action((0, 0))
    # game.do_action((1, 1))
    # game.do_action((2, 2))
    # dummy_policy = np.random.uniform(low=0, high=1, size=(9,))
    # print(game.get_legal_actions_policy_MCTS(game.board, dummy_policy))

    # todo Jump Ian Patrick Tang

    # rot90(), fliplr(), flipup()
    # we only need the original board + flipup + fliplr
    # and rot_board (k = 1) + flipup + fliplr
    # rot_board (k = 2)
    # rot_board (k = 3)
