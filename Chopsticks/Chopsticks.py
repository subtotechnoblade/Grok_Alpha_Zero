import numpy as np
from numba import njit


# Disclaimer
# I WILL NOT CONFORM to open AI's gym,
# because there isn't a way to implement monte carlo tree search without game methods
# I'll conform to open AI's gym when I implement Muzero in the future as MCTS is no longer dependent on game methods
# Mu Zero is dependent on the representation and dynamics network to get the next state called the hidden state

# this is going to be the format in which all game classes must conform to
# players are represented with -1 and 1 <- this rule cannot change
# game board must be a numpy array

# This is the default model build config and will be passed to Build_Model.py
build_config = {"embed_size": 32, # this is the vector for RWKV
          "num_heads": 2, # this must be a factor of embed_size or else an error will be raised
          "token_shift_hidden_dim": 32, # this is in the RWKV paper
          "hidden_size": None, # this uses the default 3.5 * embed size
          "num_layers": 3, # This is the total amount of RWKV layers in the model that is using
          }
# feel free to define your own build_config if you are using sth other than RWKV

train_config = {
    "total_generations": 100, # Total amount of generations, the training can stop and resume at any moment
    # a generation is defined by a round of self play, and model training

    # Self Play variables
    "games_per_generation": 100, # amount of self play games until we re train the network
    "num_explore_moves": 2,  # This is for tictactoe, a good rule of thumb is 10% to 20% of the average length of a game
    "use_gpu": True,  # Change this to false to use CPU for self play and inference
    "use_tensorrt": True,  # Assuming use_gpu is True, uses TensorrtExecutionProvider
    # change this to False to use CUDAExecutionProvider
    "num_workers": 4, # Number of multiprocessing workers used to self play

    # MCTS variables
    "MCTS_iteration_limit": 128, # The number of iterations MCTS runs for. Should be 2 to 10x the number of starting legal moves
    # set to True
    "MCTS_time_limit": None, # Not recommended to use for training
    # Set to True for a default of 30 seconds per move
    "c_puct_init": 2.5, # (shouldn't change) Exploration constant lower -> exploitation, higher -> exploration
    "dirichlet_alpha": 1.11, # should be around (10 / average moves per game) this case is (10 / 9)
    "use_njit": True, # This assumes that your check_win_MCTS uses  @njit(cache=True) or else setting this to true will cause an error

    # tensorflow training variables
    "train_epochs": 5, # The amount of epochs for training a generation's network
    "grok_lambda": 4.0, # This is for grok fast, won't be used if the model is not a Grok_Fast_EMA_Model
}
class Chopsticks:
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


    If you are using only numpy operations, feel free to decorate a static method with @njit(cache=True) for faster performance
    @njit speeds up for loops by a lot, but any numpy method can also be sped up
    the inputs must strictly be numpy arrays, bool, int, float, or tuple no list, object, or str
    -> find Brian because there are a lot of restrictions
    Example

    @staticmethod
    @njit(cache=True) # For the first call, jit will be slower but after a warmup (compile time), it should be faster
    def do_action(board, x, y, current_player):
        board[y][x] = current_player
        return board
    """

    def __init__(self):
        # MUST HAVE VARIABLES
        # define your board as a numpy array
        self.board = np.zeros(shape = (2, 3,), dtype = np.int8) # store current_player in board # 1 for the whole array

            # L R current_player
            # L R 0

        # 0 will mean that the hand is alive
        # -1 will mean that the hand is dead
        # other numbers <= 5 are the amount of fingers held up

        # current player as an int, first player will be -1 second player is 1
        self.current_player = -1
        # to swap the current the player -> current_player *= -1 (very handy)
        # action history as a list [action0, action1, ...]

        self.action_history = []
        # feel free to define more class attributes (variables)

        # action history will be recorded with the person's turn and which hand of the opponent they tap or the transfer of chopsticks
        # will be a string with the current_player added to the hand they tap "L" or "R" for index 0 and 1 accordingly
        # ex. -1 player taps the index 0 therefore recorded in the array is "-1L"
        # if it is chopstick transfer it will start with player + the hand they are transferring from + # of chopsticks transferred
        # ex. 1 player transfer 3 chopsticks to index 1. therefore what is recorded is "1R3"

        # THE COMMENTS AND EXAMPLES IN THIS CLASS IS FOR TIC TAC TOE
        # must also define policy shape
        # for tictactoe the shape of (9,) is expected because there are 9 possible moves
        # for connect 4 the shape is (7,) because there are only 7 possible moves
        # for connect 5 it is (225,) because 15 * 15 amount of actions where the index
        # you should request for a flattened policy (for tic tac toe) -> (9,) rather than (3, 3)
        # in parse policy you will have to convert the policy into actions and the associated prob_prior
        self.policy_shape = ("shape",)  # MUST HAVE or else I won't be able to define the neural network
        # just know that the illegal moves are removed in get_legal_actions_policy_MCTS() and the policy which is a probability distribution
        # is re normalized

    def get_current_player(self) -> int:
        # returns the current player
        return self.current_player

    def input_action(self):
        # returns an action using input()
        # for tictactoe it is "return int(input()), int(input())"
        pass

    def get_legal_actions(self) -> np.array:
        opposite_player = self.current_player * -1
        legal_actions_count = 0
        opposite_layer = 0
        current_layer = 1
        
        if opposite_player == 1:
            opposite_layer = 1
            current_layer = 0
        
        actions = np.zeros((1, 12))

        # tap opponents hands
        if self.board[opposite_layer, 0] != -1:
            legal_actions_count += 1
            actions[0, legal_actions_count] = self.current_player + "L"
        if self.board[opposite_layer, 1] != -1:
            legal_actions_count += 1
            actions[0, legal_actions_count] = self.current_player + "R"

        # transfer chopsticks
        left_hand = self.board[current_layer, 0] 
        right_hand = self.board[current_layer, 1]

        left_hand_copy = left_hand
        right_hand_copy = right_hand

        # for left_hand_copy < 5 :

    

            

        # max 12 possible ways as you can tap opponents two hands or transfer chopsticks between current player's hands (5 max)


        self.current_player
        # returns the all possible legal actions in a numpy array [action1, action2, ..., actionN] given self.board
        # Note that this action will be passed into do_action() and do_action_MCTS
        # MAKE SURE THERE are no duplicates (pretty self explanatory)
        pass




        # actions = []
        # check if split is possible
        # [-1, 2, 3]
        # [ 0, 1, 4]




        # if sum(self.board[1][1:]) == 4:
        #     # [1, 3] -> [2, 2]
        # elif sum(self.board[1][1:]) == 5:

            # [1, 4] -> [2, 3]
            # [1, 4] -> [3, 2]

        # check if split is possible
        #
        return self.get_legal_actions_MCTS(self.board)

    @staticmethod
    # @njit(cache=True)
    def get_legal_actions_MCTS(board):
        # this returns the legal moves, just do the same thing as get_legal_actions
        pass

    @staticmethod
    # @njit(cache=True)
    def get_legal_actions_policy_MCTS(board: np.array, policy: np.array, shuffle=False) -> (np.array, np.array):
        """
        THIS PART IS REALLY HARD FOR BEGINNERS, I RECOMMEND TO SKIP THIS PART UNTIL YOU ARE MORE CONFIDENT
        :param board: numpy array of the board
        :param policy: a numpy array of shape = self.policy shape defined in __init__, straight from the neural network's policy head
        :param shuffle: You might want to shuffle the policy and legal_actions because the last index is where the search starts
        if it is too hard you don't implement anything much will happen, its just that randomizing might improve convergence just by a but
        :return: legal_actions as a list [action0, action1, ...], child_prob_prior as a numpy array

        In essence, index 0 in legal_actions and child_prob_prior should be the probability of the best move for that legal action
        so for example in tic tac toe
        legal_actions =             [0,     1,   2,   3,    4,    5,    6,    7,    8,    9]
        child_prob_prior / policy = [0.1, 0.1, 0.3, 0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05] (I made these up, shape=(9,))
        it means that action 0 will have a predicted chance of 10% of being the best move, 1 will have 10%, 2 will have 30% and so on
        Note that this is a probability distribution, after removing the un wanted actions you have to normalize it to sum up to 1
        do that with policy /= np.sum(policy)


        Anyway, if your self.policy shape matches the action shape you can just return the policy without doing anything
        recommend to use numba njit to speed up this method for MCTS
        """

        # example with randomization for tictactoe:
        # assuming policy from the comments above
        # board = board.reshape((-1)) flatten the 3 by 3 board into a length 9 array
        # policy = policy.reshape(-1) flatten the policy
        # policy = policy[board == 0] keep the values that correspond to an empty part of the board
        # policy /= np.sum(policy) # normalize it
        # legal_actions = np.argwhere(board == 0) # get the indexes where the board is empty
        # if shuffle:
        # shuffled_indexes = np.random.permutation(len(legal_actions)) # create random indexes
        # legal_actions, policy = legal_actions[shuffled_indexes], policy[shuffled_indexes] # index the arrays to shuffled them
        # return legal_moves, policy

        # MAKE SURE THERE ARE NO DUPLICATES because it is going to increase the tree complexity thus slowing things down
        # it is also going to give a wrong policy and weird errors might occur
        # for checking use
        # assert len(legal_actions) == len(set(legal_action))
        # set cannot have any duplicates and thus removed so if the lengths are different the there is a problem
        pass

    def do_action(self, action) -> np.array:
        # places the move onto the board
        pass

    @staticmethod
    # @njit(cache=True)
    def do_action_MCTS(board, action, current_player) -> np.array:
        # this is for the monte carlo tree search's
        pass

    def get_input_state(self) -> np.array:
        # gets the numpy array for the neural network
        # for now just return the board as a numpy array
        # Brian will probably implement this later for specific neural networks
        # RWKV can just take in the board without problem
        # the original alphazero's network required the past boards
        # Uses in the root node of MCTS
        pass

    @staticmethod
    # @njit(cache=True)
    def get_input_state_MCTS(board) -> np.array:
        # Used for retrieving the state for any child nodes (not the root)
        # just return the board from the inputs because board is alo an np.array
        return board

    def check_win(self) -> int:
        # returns the player who won (-1 or 1), returns 0 if a draw is applicable
        # return -2 if no player has won / the game hasn't ended

        # feel free to implement the logic in check_win_MCTS
        # and call it like Game.check_win_MCTS(self.board, self.action_history[-1], self.current_player)
        # of course "Game" should be changed to the class name, "Game" is just an example
        pass

    @staticmethod
    # @njit(cache=True)
    def check_win_MCTS(board, last_action, current_player) -> int:
        # Used to check if MCTS has reached a terminal node
        # return the player (-1 or 1) if that player won, 0 if draw, and -2 if game hasn't ended
        pass

    def compute_policy_improvement(self, statistics):
        # given [[action, probability], ...] compute the new policy which should be of shape=self.policy_shape
        # example for tic tac toe statistics=[[[0, 0], 0.1], [[1, 0], 0.2], ...] as in [[action0, probability for action0], ...]
        # you should return a board with each probability assigned to each move
        # return [0.1, 0.2, ...]
        # note that the coordinate [0, 0] corresponds to index 0 in the flattened board
        # this should map the action and probability to a probability distribution
        pass
    @staticmethod
    #@njit(cache=True)
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

        return [board], [policy] # just return [board], [policy] if you don't want to implement this
        # don't be lazy, this will help convergence very much



if __name__ == "__main__":
    # from Game_Tester import Game_Tester
    # from Grok_Alpha_Zero.Game_Tester import Game_Tester

    # game = Chopsticks()

    import os
    import Game_Tester
    print(os.listdir())