import numpy as np
from numba import njit
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

#todo functions: make a place move function that takes:
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

build_config = {"embed_size": 64, # this is the vector for RWKV
                "num_heads": 1, # this must be a factor of embed_size or else an error will be raised
                "token_shift_hidden_dim": 32, # this is in the RWKV paper
                "hidden_size": None, # None uses the default 3.5 * embed, factor for upscaling in channel mix
                "num_layers": 3, # This is the total amount of RWKV layers in the model that are used

                "use_stable_max": True,
                "use_grok_fast": True,
                "use_orthograd": True,
                "grok_lambda": 4.5,  # This is for grok fast, won't be used if model is Grok_Fast_EMA_Model
          }

train_config = {
    "total_generations": 100, # Total number of generations, the training can be stopped and resume at any moment
    # a generation is defined by a round of self play, padding the dataset, model training, converting to onnx

    # Self Play variables
    "games_per_generation": 10, # number of self play games until we re train the network
    "max_actions": 9, # Note that this should be
    "num_explore_actions": 1,  # This is for tictactoe, a good rule of thumb is 10% to 20% of the average length of a game
    "use_gpu": False,  # Change this to false to use CPU for self play and inference
    "use_tensorrt": False,  # Assuming use_gpu is True, uses TensorrtExecutionProvider
    # change this to False to use CUDAExecutionProvider
    "num_workers": 6, # Number of multiprocessing workers used to self play

    # MCTS variables
    "MCTS_iteration_limit": 128, # The number of iterations MCTS runs for. Should be 2 to 10x the number of starting legal moves
    # True defaults to iteration_limit = 3 * len(starting legal actions)
    "MCTS_time_limit": None,  # Not recommended to use for training, True defaults to 30 seconds
    "c_puct_init": 2.5, # (shouldn't change) Exploration constant lower -> exploitation, higher -> exploration
    "dirichlet_alpha": 1.11, # should be around (10 / average moves per game)
    "use_njit": True, # This assumes that your check_win_MCTS uses  @njit(cache=True) or else setting this to true will cause an error

    "num_previous_generations": 3, # The previous generation's data that will be used in training
    "train_percent": 1.0, # The percent used for training after the test set is taken
    "train_decay": 0.75, # The decay rate for previous generations of data previous_train_percent = current_train_percent * train_decay
    "test_percent": 0.1, # The percent of a dataset that will be used for validation
    "test_decay": 0.75, # The decay rate for previous generations of data previous_test_percent = current_test_percent * test_decay

    "train_batch_size": 64, # The number of samples in a batch for training in parallel
    "test_batch_size": None, # If none, then train_batch_size will be used for the test batch size
    "learning_rate": 1e-3, # Depending on how many RWKV blocks you use. Recommended to be between 1e-3 to 5e-4
    "decay_lr": 0.1,  # When the generation reaches 10%, 20% ,... learning rate will be decreased linearly
    "beta_1": 0.9, # DO NOT TOUCH unless you know what you are doing
    "beta_2": 0.989, # DO NOT TOUCH. This determines whether it groks or not. Hovers between 0.985 to 0.995
    "train_epochs": 5, # The number of epochs for training
}


class TicTacToe:
    def __init__(self):
        self.current_player = -1
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.action_history = []
        # doesn't return anything

        self.policy_shape = (9,)

    def get_current_player(self):
        return self.current_player

    def input_action(self):
        while True:
            try:
                coords = input("Move:").split(" ")
                coords[0], coords[1] = int(coords[0]), int(coords[1])
                return np.array(coords)
            except:
                print("Invalid Move")
                continue

            x, y = coords
            if (0 > x > 2 or 0 < y > 2) or self.board[coords[1]][coords[0]] != 0:
                print("Illegal move given")
                continue



    def get_legal_actions(self): #-> list[tuple[int, int], ...] or np.array:
        """
        self.total_action is a set
        :return: a list of actions [action0, action1]
        """
        # return np.array(list(self.total_actions))
        return self.get_legal_actions_MCTS(self.board, self.get_current_player(), np.array(self.action_history))

    @staticmethod
    @njit(cache=True)
    def get_legal_actions_MCTS(board, current_player, action_history):
        return np.argwhere(board == 0)[:,::-1]

    @staticmethod
    # @njit(cache=True)
    def get_legal_actions_policy_MCTS(board: np.array,
                                      current_player:int,
                                      action_history: np.array,
                                      policy: np.array,
                                      shuffle: bool=False):

        legal_actions = np.argwhere(board == 0)[:, ::-1]
        legal_policy = policy[board.reshape(-1) == 0]
        legal_policy /= np.sum(legal_policy)

        if shuffle:
            random_indexes = np.random.permutation(len(legal_actions))
            legal_actions = legal_actions[random_indexes]
            legal_policy = legal_policy[random_indexes]

        return legal_actions, legal_policy

    def do_action(self, action): # must conform to this format - Brian
        x, y = action
        self.board[y][x] = self.current_player
        self.current_player = self.current_player * -1
        self.action_history.append(action)


    @staticmethod
    @njit(cache=True)
    def do_action_MCTS(board, action, current_player):
        x, y = action
        board[y][x] = current_player
        return board

    def get_input_state(self):
        return self.board

    @staticmethod
    # @njit(cache=True)
    def get_input_state_MCTS(board):
        # Used for retrieving the state for any child nodes (not the root)
        # just return the board from the inputs
        return board

    def _check_row(self, row): #A function for checking if there's a win on each row.
        if row[0] != 0 and row[0] == row[1] == row[2]:
            return True #If all 3 items on a row equal, someone has won. Yippee!
        return False #If not then sadly not :(

    def check_win(self):
        # Note that self.current player is actually the next player
        # -self.current_player because we've just played a move which changes to the next player
        # thus if you found a win you have to return the previous player since that player just played -Brian
        for row in self.board:
            if self._check_row(row) is True: #Pulling out func from LALA-land
                "                    ^ Jessy, just letting you know that 'is True' is not necessary for python"
                "It's equally valid to do 'if self._check_row(row):' - Brian"
                return -self.current_player #Someone has won! :D

        for column_index in range(3): #Checking if someone has won in columns
            column = self.board[:, column_index]
            if self._check_row(column) is True: #Pulling out func from LALA-land pt2
                return -self.current_player #Someone has won, yippee :3

        diag1 = np.diag(self.board) #Checking if someone has won in diagonals
        if self._check_row(diag1) is True: #Pulling out func from dark magic
            return -self.current_player #Someone has won :P

        diag2 = np.diag(np.fliplr(self.board))
        if self._check_row(diag2) is True: #Pulling func from Narnia
            return -self.current_player #Someone has won :)

        flattened_board = self.board.reshape((9,)) #Reshape board into 9 items, this is to check for draw
        for value in flattened_board:
            if value == 0: #Still empty spaces on the board, stop checking!!
                break
        else:
            return 0 # return a draw if break never happens - Brian

        return -2 #Game continues :3

    @staticmethod
    @njit(cache=True)
    def check_win_MCTS(board, action_history, current_player):
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
        new_policy = np.zeros(self.policy_shape, np.float32)
        for (x, y), prob in statistics:
            new_policy[x + 3 * y] = prob

        return new_policy

    @staticmethod
    @njit(cache=True)
    def augment_sample_fn(boards: np.array, policies: np.array):
        policies = policies.reshape((-1, 3, 3))# we need
        # to reshape this because we can only rotate a matrix, not a vector

        augmented_boards = []
        augmented_policies = []

        for action_id in range(boards.shape[0]):
            board = boards[action_id]
            # augmented_board = np.zeros((8, board_shape))
            augmented_board = [board, np.flipud(board), np.fliplr(board)]


            policy = policies[action_id]
            augmented_policy = [policy, np.flipud(policy), np.fliplr(policy)]


            for k in range(1, 4):
                rot_board = np.rot90(board, k)
                augmented_board.append(rot_board)

                rot_policy = np.rot90(policy, k)
                augmented_policy.append(rot_policy)

                if k == 1:
                    augmented_board.append(np.flipud(rot_board))
                    augmented_board.append(np.fliplr(rot_board))

                    augmented_policy.append(np.flipud(rot_policy))
                    augmented_policy.append(np.fliplr(rot_policy))
            augmented_boards.append(augmented_board)

            augmented_policies.append(augmented_policy)
        return augmented_boards, augmented_policies

    def augment_sample(self, boards, policies):
        # Note that values don't have to be augmented since they are the same regardless of how a board is rotated
        augmented_boards, augmented_policies = self.augment_sample_fn(boards, policies)
        return np.array(augmented_boards, dtype=boards[0].dtype).transpose([1, 0, 2, 3]), np.array(augmented_policies, dtype=np.float32).reshape((-1, 8, 9)).transpose([1, 0, 2])



if __name__ == "__main__":
    # test your code here
    game = TicTacToe()
    # print(game.board)
    # board = np.zeros((2, 2))
    # board = np.array([0.0, 0.0], dtype=np.uint8)
    # print(board.dtype)

    #dtype -> data type
    # integer
    # np.uint8 0 to 255
    # np.int16 short, int
    # np.int32 long
    # np.int64 long long






    #








    # game.do_action((0, 0))
    # game.do_action((1, 1))
    # game.do_action((2, 2))
    dummy_policy = np.random.uniform(low=0, high=1, size=(9,))
    # print(game.get_legal_actions_policy_MCTS(game.board, dummy_policy))












    # todo Jump Ian Patrick Tang




    # rot90(), fliplr(), flipup()
    # we only need the original board + flipup + fliplr
    # and rot_board (k = 1) + flipup + fliplr
    # rot_board (k = 2)
    # rot_board (k = 3)