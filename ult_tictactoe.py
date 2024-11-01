import numpy as np
from numba import njit

class UltimateTicTacToe:
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
        self.board = np.array([])
        # current player as an int, first player will be -1 second player is 1
        self.current_player = -1
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
        self.policy_shape = ("your policy shape (length)",) # MUST HAVE or else I won't be able to define the neural network
        # just know that the illegal moves are removed and the policy which is a probability distribution
        # is re normalized

    def get_current_player(self):
        # returns the current player
        pass

    def get_legal_actions(self):
        # returns the all possible legal actions in a list [action1, action2, ..., actionN] given self.board
        # Note that this action will be passed into do_action() and do_action_MCTS
        pass
    @staticmethod
    # @njit(cache=True)
    def get_legal_actions_policy_MCTS(board: np.array, policy: np.array, shuffle=False):
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

        # example with randomization for tictac toe:
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
        pass


    def do_action(self, action):
        # places the move onto the board
        pass
    @staticmethod
    # @njit(cache=True)
    def do_action_MCTS(board, action, current_player):
        # this is for the monte carlo tree search's
        pass

    def get_state(self):
        # gets the numpy array for the neural network
        # for now just return the board as a numpy array
        # Brian will probably implement this later for specific neural networks
        # RWKV can just take in the board without problem
        # the original alphazero's network required the past boards
        # Uses in the root node of MCTS
        pass

    @staticmethod
    # @njit(cache=True)
    def get_state_MCTS(board):
        # Used for retrieving the state for any child nodes (not the root)
        # just return the board from the inputs
        return board

    def check_win(self):
        # returns the player who won (-1 or 1), returns 0 if a draw is applicable
        # return -2 if no player has won / the game hasn't ended
        pass
    @staticmethod
    # @njit(cache=True)
    def check_win_MCTS(board, last_action):
        #Used to check if MCTS has reached a terminal node
        pass
    @staticmethod
    @njit(cache=True)
    def get_winning_actions_MCTS(board, current_player, fast_check=False):
        # Brian will be looking very closely at this code when u implement this
        # reocmment to use check_win_MCTS unless there is a more efficient way
        # making sure that this doesn't slow this MCTS to a halt
        # if your game in every case only has 1 winning move you don'y have to use fast_check param
        # please do not remove the fast_check parameter
        # check the gomoku example for more info
        pass


