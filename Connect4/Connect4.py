import numpy as np
from numba import njit

build_config = {"num_resnet_layers": 2,  # This is the total amount of resnet layers in the model that are used
                "num_filters": 128,
                "use_stablemax": True,  # use stablemax, which will also use stablemax crossentropy
                "use_grok_fast": False,  # from grokfast paper
                "use_orthograd": False,  # from grokking at the edge of numerica stability
                "grok_lambda": 4.5,  # This is for grok fast, won't be used if model is Grok_Fast_EMA_Model
          }
train_config = {
    "total_generations": 100,  # Total amount of generations, the training can be stopped and resume at any moment
    # a generation is defined by a round of self play, padding the dataset, model training, converting to onnx

    # Self Play variables
    "games_per_generation": 100,  # amount of self play games until we re train the network
    "max_actions": 42,  # Note that this should be less than max actions,
    "num_explore_actions_first": 8,  # A good rule of thumb is how long the opening should be for player -1
    "num_explore_actions_second": 7,  # Since player 1 is always at a disadvantage, we explore less and attempt to play better moves

    "use_gpu": True,  # Change this to False to use CPU for self play and inference
    "use_tensorrt": True,  # Assuming use_gpu is True, uses TensorrtExecutionProvider
    # change this to False to use CUDAExecutionProvider
    "use_inference_server": True,  # if an extremely large model is used, because of memory constraints, set this to True
    "max_cache_depth": 1,  # maximum depth in the search of the neural networks outputs we should cache, use this if the inference speed is under 1000it/s
    "num_workers": 8,  # Number of multiprocessing workers used to self play

    # MCTS variables
    "MCTS_iteration_limit": 100,  # The number of iterations MCTS runs for. Should be 2 to 10x the number of starting legal moves
    # True defaults to iteration_limit = 3 * len(starting legal actions)
    "MCTS_time_limit": None,  # Not recommended to use for training, True defaults to 30 seconds
    "use_njit": None,  # None will automatically infer what is supposed to be use for windows/linux

    "use_gumbel": False,  # use gumbel according to https://openreview.net/pdf?id=bERaNdoegnO
    # These params will only be used when use_gumbel is set to True
    "m": 7,  # Number of actions sampled in the first stage of sequential halving
    "c_visit": 50.0,
    "c_scale": 1.0,

    # These params will be used when use_gumbel is set to False
    "c_puct_init": 2.5,  # (shouldn't change) Exploration constant lower -> exploitation, higher -> exploration
    "dirichlet_alpha": 0.5,  # should be around (10 / average moves per given position)

    # "opening_actions": [[3, 0.5]], # starting first move in the format [[action1, prob0], [action1, prob1], ...],
    # if prob doesn't add up to 1, then the remaining prob is for the MCTS move

    "num_previous_generations": 3,  # The previous generation's data that will be used in training
    "train_percent": 1.0,  # The percent used fr training after the test set is taken
    "train_decay": 0.8,  # The decay rate for previous generations of data previous_train_percent = current_train_percent * train_decay
    "test_percent": 0.1,  # The percent of a dataset that will be used for validation
    "test_decay": 0.33,  # The decay rate for previous generations of data previous_test_percent = current_test_percent * test_decay

    "mixed_precision": None,  # None for no mixed precision, mixed_float16 for float16
    "train_batch_size": 1024,  # The number of samples in a batch for training in parallel
    "test_batch_size": None,  # If none, then train_batch_size will be used for the test batch size
    "gradient_accumulation_steps": None,
    "learning_rate": 3e-4,  # Depending on how many layers you use. Recommended to be between 5e-4 to 1e-5 or even lower
    "decay_lr_after": 15,  # When the n generations pass,... learning rate will be decreased by lr_decay
    "lr_decay": 0.75,  # multiplies this to learning rate every decay_lr_after
    "beta_1": 0.9,  # DO NOT TOUCH unless you know what you are doing
    "beta_2": 0.995,  # DO NOT TOUCH. This determines whether it groks or not. Hovers between 0.985 to 0.995
    "optimizer": "Nadam",  # optimizer options are ["Adam", "AdamW", "Nadam"]
    "train_epochs": 7,  # The number of epochs for training
}
# # resources
# print("Please read the comments before you start this project")
# print("Some of the examples are printed out, so run this file to see them")
# # differences between lists and arrays.
# # Arrays must be homogenous, meaning that it can only have numbers and defined dimension length
# # examples:
#     # array -> [1, 2, 3] # shape = (3,)
#     # not an array (list) -> [1, 2, 3, "3"] # note that "3" is a string
#     # not an array (list) -> [1, 2, 3, [3, 3]] # note that [3, 3] is another dimension, but not consistent with 1 or 2 or 3
#     # array -> [[1, 2],
#     #           [2, 3],
#     #           [3, 4]] # dimensionality is consistent, shape=(3, 2,)
#     # not an array -> [[1, 2, 3]]
# #                      [4, 5, 6]
# #                      [7, 8]] # this should have 3 numbers but it only has 2
#
#
# # quick refresher on shapes of arrays
# # note that arrays has a shape while lists do not
# # shape defines the amount of elements within that dimension
# #
# # an array of shape (3,) -> [1, 1, 1], called a vector
# # an array of shape (1, 3,) -> [[1, 1, 1]] # called a matrix because it has 2 dimensions
# # an array of shape (3, 1,) -> [[1],
# #                               [1],
# #                               [1]] # matrix
# # an array of shape (3, 3) -> [[1, 1, 1],
# #                              [1, 1, 1],
# #                              [1, 1, 1]]
# # explanation: there are three rows within the first dimension and 3 numbers in each row
# # ^ perfect for tic tac toe
#
# # example of creating a zero's matrix
# print("3 by 3 matrix")
# print(np.zeros((3, 3)))
# print("4 by 4 matrix")
# print(np.zeros((4, 4)))
#
# print("\nIndexing")
# # intro to indexing
# # propose we make an array
#                  #[[1, 2, 3],
#                  # [4, 5, 6],
#                  # [7, 8, 9]]
# arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# # or
# arr = np.arange(start=1, stop=10).reshape((3, 3)) # both do the same thing
# print(arr)
# # getting the first row:
# print("First row:")
# first_row = arr[0]
# print(first_row)
# # getting the second row:
# print("Second_row")
# second_row = arr[1]
# print(second_row)
# # getting the third row:
# print("Third row")
# third_row = arr[2] # this is the last row
# print(third_row)
#
# print("Indexing a number from arr")
# print("first element from first row")
# first_element = arr[0][0] # 0->2 and do on, note that these are indexes that start from 0
# print(first_element)
# # or alternatively first_element = arr[0, 0]
# # Note that because there are 2 dimensions to get a number which is in the last dimension, we must
# # access that dimension by indexing 2 times, [0][0]
# # in a 3-dimensional 3D array, you have to index 3 times [0][0][0] to get to a number
#
#
# print("\nManipulating Arrays")
# # You can change the value of an array/row/dimension with
# arr = np.zeros((3, 3))
#
# print("change a single element")
#
# arr[0][0] = 1
# print(arr)
#
# print("change whole row")
# arr = np.zeros((3, 3))
# arr[1] = 1
# print(arr)
#
# print("change whole array")
# arr = np.zeros((3, 3))
# arr[:] = 1
# print(arr)
#
#
# print("\nArithmetic operations with arrays")
# # similarly to manipulating arrays
# # you can:
# print("Multiply only the 3rd element of the first row")
# arr = np.ones((3, 3))
# arr[2][2] *= 2
# print(arr)
#
# print("Multiplying only the second row")
# arr = np.ones((3, 3))
# arr[1] *= 2
# print(arr)
#
# print("Multiplying the whole array")
# arr = np.ones((3, 3))
# arr *= 3
# print(arr)
# print("The operations can be any operations such as +, -, *, /, %, //")
#
# # creating a numpy array from a list -> arr = np.array([1, 2, 3])
# # creating a numpy zeros array -> zeros_arr = np.zeros((3, 3))
#
# # optimizations to consider
# # use integer 8 [-128, 127] rather than float 32 [-3.40282347E+38 to -1.17549435E-38] to save memory
# # (8 bits compared to 32 bits)
# # np.zeros((3, 3), dtype=np.int8) # dtype means data type
# # list of dtypes: [np.bool_, np.int8, np.int16, np.int32, np.int64, np.float16, np.float64]

class Connect4:
    # refer to Guide.py to implement each method, please be careful of what is used and returned
    # if you want pure speed recommend the decorator @njit for methods with for loops and numpy operations
    # contact Brian if you want more info

    # DO NOT INHERIT THIS CLASS and overwrite the methods, it's a waste of memory, just copy and implement each method
    """
    Any @staticmethods are for the tree search algorithm as it is memory intensive to store the game class and use the methods
    Note that numba's @njit can only be used with @staticmethods, because numba cannot work with any internal class methods
    @staticmethod just creates a method that isn't associated with a class, its just for organization

    Example
    def foo():
        return 0

    class Bar:
        @staticmethod
        def foo():
            return 0

    foo() and Bar.foo() are the same thing, just that Bar.foo() is organized to be part of the Bar class


    If you are using only numpy operations, feel free to decorate a static method with @njit for faster performance
    @njit speeds up for loops by a lot, but any numpy method can also be sped up
    the inputs must strictly be numpy arrays, bool, int, float, or tuple no list, object, or str
    -> find Brian because there are a lot of restrictions
    Example

    @staticmethod
    @njit # For the first call, jit will be slower but after a warmup (compile time), it should be faster
    def do_action(board, x, y, current_player):
        board[y][x] = current_player
        return board
    """

    def __init__(self):
        # MUST HAVE VARIABLES
        # define your board as a numpy array
        self.board = np.zeros((6, 7), dtype=np.int8)  # should i use np.array?
        # current player as an int, first player will be -1 second player is 1
        self.next_player = -1
        # The DEFINITION of next_player is the player that is going to play but hasn't put their move on the board
        # The DEFINITION of current_player is the player that has just played and their move is on the board

        # to swap the current the player -> current_player *= -1 (very handy)
        # action history as a list [action0, action1, ...]
        self.action_history = []
        # feel free to define more class attributes (variables)

        # THE COMMENTS AND EXAMPLES IN THIS CLASS IS FOR TIC TAC TOE
        # must also define policy shape
        # for tictactoe the shape of (9,) is expected because there are 9 possible moves
        # for connect 4 the shape is (7,) because there are only 7 possible moves
        # for connect 5 it is (225,) because 15 * 15 amount of actions where the index
        # you should request for a flattened policy (for tic tac toe) -> (9,) rather than (3, 3)
        # in parse policy you will have to convert the policy into actions and the associated prob_prior
        self.policy_shape = (7,)
        # MUST HAVE or else I won't be able to define the neural network
        # just know that the illegal moves are removed and the policy which is a probability distribution
        # is re normalized

    def get_next_player(self):
        # returns the next player
        return self.next_player

    def input_action(self):
        # returns an action using input()
        # for tictactoe it is "return int(input()), int(input())"
        while True:
            try:
                action = int(input("Action: "))
                if np.sum(abs(self.board[:, action])) < 6:
                    return action
                else:
                    print("Illegal move")
            except:
                print("Try again")

    def get_legal_actions(self):
        # returns the all possible legal actions in a list [action1, action2, ..., actionN] given self.board
        # Note that this action will be passed into do_action() and do_action_MCTS
        # MAKE SURE THERE are no duplicates (pretty self explanatory)
        return self.get_legal_actions_MCTS(self.board, self.next_player, np.array(self.action_history))

    @staticmethod
    @njit(cache=True)
    def get_legal_actions_MCTS(board: np.array, next_player: int, action_history: np.array):
        legal_actions = []
        for x in range(7):
            if np.sum(np.abs(board[:, x])) < 6:
                legal_actions.append(x)
        return np.array(legal_actions, dtype=np.int8)

    @staticmethod
    @njit(cache=True)
    def get_legal_actions_policy_MCTS(board: np.array,
                                      current_player: int,
                                      action_history: np.array,
                                      policy: np.array,
                                      normalize=True,
                                      shuffle=False) -> (np.array, np.array):
        legal_actions = []
        for x in range(7):
            if np.sum(np.abs(board[:, x])) < 6:
                legal_actions.append(x)
        legal_actions = np.array(legal_actions, dtype=np.int8)

        legal_policy = policy[legal_actions]
        if normalize:
            legal_policy /= np.sum(legal_policy)

        # if shuffle:
        #     random_shuffle = np.random.permutation(len(legal_actions))
        #     return legal_actions[random_shuffle], legal_policy[random_shuffle]
        return legal_actions, legal_policy

    def do_action(self, action):
        # places the move onto the board
        action_col = self.board[:, action]
        max_index = 5 - np.sum(np.abs(action_col))
        self.board[max_index][action] = self.next_player
        self.next_player *= -1
        self.action_history.append(action)

    @staticmethod
    @njit(cache=True)
    def do_action_MCTS(board, action, next_player):
        # this is for the monte carlo tree search's
        action_col = board[:, action]
        max_index = 5 - np.sum(np.abs(action_col))
        board[max_index][action] = next_player
        return board

    def get_input_state(self):
        # gets the numpy array for the neural network
        # for now just return the board as a numpy array
        # Brian will probably implement this later for specific neural networks
        # RWKV can just take in the board without problem
        # the original alphazero's network required the past boards
        # Uses in the root node of MCTS
        return self.get_input_state_MCTS(self.board, -self.next_player, np.array(self.action_history, dtype=np.int8))

    @staticmethod
    @njit(cache=True)
    def get_input_state_MCTS(board: np.array, current_player: int, action_history: np.array) -> np.array:
        # Used for retrieving the state for any child nodes (not the root)
        # just return the board from the inputs
        board_state = np.zeros((4, 6, 7), dtype=np.int8)
        board_state[0] = current_player
        board_state[-1] = board

        max_length = len(action_history) - 1
        max_length = min(max_length, 3)

        prev_board = board.copy()
        for i in range(-1, -max_length - 1, -1):
            x = int(action_history[i])
            y = min(np.where(prev_board[:, x] != 0))[0]

            prev_board[y][x] = 0
            board_state[i - 1] = prev_board
        return np.transpose(board_state, (1, 2, 0))

    def check_win(self) -> bool:
        return self.check_win_MCTS(self.board, -self.next_player, np.array(self.action_history))

    @staticmethod
    @njit(cache=True)
    def check_win_MCTS(board, current_player, action_history):
        # all in computer coordinates
        x = action_history[-1]


        column = board[:, x]
        y = min(*np.where(column == current_player))
        row = board[y]
        # print(x, y)

        start_x = max(0, x - 3)
        end_x = min(6, x + 3)

        # Vertical range (ensure within 0-5)
        start_y = min(5, y + 3)  # Farthest down
        end_y = max(0, y - 3)  # Farthest up


        # horizonal check
        count = 0
        for i in range(start_x, end_x + 1):
            if row[i] == current_player:
                count += 1
                if count == 4:
                    return current_player
            else:
                count = 0

        count = 0
        for i in range(start_y, end_y - 1, -1):
            if column[i] == current_player:
                count += 1
                if count == 4:
                    return current_player
            else:
                count = 0

        count_d_bl_tr = 0
        for i in range(-min(x - start_x, start_y - y), min(end_x - x, y - end_y) + 1):
            if board[y - i, x + i] == current_player:
                count_d_bl_tr += 1
                if count_d_bl_tr == 4:
                    return current_player
            else:
                count_d_bl_tr = 0

        count_d_tl_br = 0
        for i in range(-min(x - start_x, y - end_y), min(end_x - x, start_y - y) + 1):
            if board[y + i, x + i] == current_player:
                count_d_tl_br += 1
                if count_d_tl_br == 4:
                    return current_player
            else:
                count_d_tl_br = 0

        if np.all(board != 0):
            return 0

        return -2


    def compute_policy_improvement(self, statistics):
        # given [[action, probability], ...] compute the new policy which should be of shape=self.policy_shape
        # example for tic tac toe statistics=[[[0, 0], 0.1], [[1, 0], 0.2], ...] as in [[action0, probability for action0], ...]
        # you should return a board with each probability assigned to each move
        # return [0.1, 0.2, ...]
        # note that the coordinate [0, 0] corresponds to index 0 in the flattened board
        # this should map the action and probability to a probability distribution
        policy = np.zeros(7, dtype=np.float32)
        for action, prob in statistics:
            policy[action] = prob
        return policy

    @staticmethod
    @njit(cache=True)
    def augment_sample(board, policy):
        # optional method to improve convergence
        # rotate the board and flip it using numpy and return those as a list along with the original
        # remember to rotate the flip the policy in the same way as board
        # return [board, rotated_board, ...], [policy, rotated_policy, ...]

        # Note the optimal rotations and flips for tictactoe, and gomoku is
        # [original arr, flipup(arr), fliplr(arr)]
        # and [np.rot_90(arr, k = 1) + flipup(rot_arr), fliplr(rot_arr)]
        # and [np.rot_90(arr, k = 2)
        # and [np.rot_90(arr, k = 3)

        # Note that this won't be the case for connect4, and ult_tictactoe

        augmented_boards = np.stack((board, np.fliplr(board)))
        augmented_polices = np.stack((policy, np.fliplr(policy)))

        return augmented_boards, augmented_polices  # just return [board], [policy] if you don't want to implement this
        # don't be lazy, this will help convergence very much


if __name__ == "__main__":
    from Game_Tester import Game_Tester

    # tester = Game_Tester(Connect4)
    # tester.test()
    game = Connect4()
    # action = 0
    game.do_action(0)
    game.do_action(0)
    game.do_action(0)
    game.get_input_state()

    # game.board[6][1] = -1
    # game.board[5][1] = -1
    # game.board[4][2] = -1
    # game.board[3][3] = -1

    # game.board[5][2] = -1
    # game.board[4][2] = -1
    # game.board[3][2] = -1
    # game.board[2][2] = -1
    #
    # game.do_action(2)
    # # game.action_history.append(6)
    #
    # print(game.board)
    # print(game.check_win())
