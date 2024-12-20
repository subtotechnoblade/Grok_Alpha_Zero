import numpy as np
from numba import njit
# resources
print("Please read the comments before you start this project")
print("Some of the examples are printed out, so run this file to see them")
# differences between lists and arrays.
# Arrays must be homogenous, meaning that it can only have numbers and defined dimension length
# examples:
    # array -> [1, 2, 3] # shape = (3,)
    # not an array (list) -> [1, 2, 3, "3"] # note that "3" is a string
    # not an array (list) -> [1, 2, 3, [3, 3]] # note that [3, 3] is another dimension, but not consistent with 1 or 2 or 3
    # array -> [[1, 2],
    #           [2, 3],
    #           [3, 4]] # dimensionality is consistent, shape=(3, 2,)
    # not an array -> [[1, 2, 3]]
#                      [4, 5, 6]
#                      [7, 8]] # this should have 3 numbers but it only has 2


# quick refresher on shapes of arrays
# note that arrays has a shape while lists do not
# shape defines the amount of elements within that dimension
#
# an array of shape (3,) -> [1, 1, 1], called a vector
# an array of shape (1, 3,) -> [[1, 1, 1]] # called a matrix because it has 2 dimensions
# an array of shape (3, 1,) -> [[1],
#                               [1],
#                               [1]] # matrix
# an array of shape (3, 3) -> [[1, 1, 1],
#                              [1, 1, 1],
#                              [1, 1, 1]]
# explanation: there are three rows within the first dimension and 3 numbers in each row
# ^ perfect for tic tac toe

# example of creating a zero's matrix
print("3 by 3 matrix")
print(np.zeros((3, 3)))
print("4 by 4 matrix")
print(np.zeros((4, 4)))

print("\nIndexing")
# intro to indexing
# propose we make an array
                 #[[1, 2, 3],
                 # [4, 5, 6],
                 # [7, 8, 9]]
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# or
arr = np.arange(start=1, stop=10).reshape((3, 3)) # both do the same thing
print(arr)
# getting the first row:
print("First row:")
first_row = arr[0]
print(first_row)
# getting the second row:
print("Second_row")
second_row = arr[1]
print(second_row)
# getting the third row:
print("Third row")
third_row = arr[2] # this is the last row
print(third_row)

print("Indexing a number from arr")
print("first element from first row")
first_element = arr[0][0] # 0->2 and do on, note that these are indexes that start from 0
print(first_element)
# or alternatively first_element = arr[0, 0]
# Note that because there are 2 dimensions to get a number which is in the last dimension, we must
# access that dimension by indexing 2 times, [0][0]
# in a 3-dimensional 3D array, you have to index 3 times [0][0][0] to get to a number


print("\nManipulating Arrays")
# You can change the value of an array/row/dimension with
arr = np.zeros((3, 3))

print("change a single element")

arr[0][0] = 1
print(arr)

print("change whole row")
arr = np.zeros((3, 3))
arr[1] = 1
print(arr)

print("change whole array")
arr = np.zeros((3, 3))
arr[:] = 1
print(arr)


print("\nArithmetic operations with arrays")
# similarly to manipulating arrays
# you can:
print("Multiply only the 3rd element of the first row")
arr = np.ones((3, 3))
arr[2][2] *= 2
print(arr)

print("Multiplying only the second row")
arr = np.ones((3, 3))
arr[1] *= 2
print(arr)

print("Multiplying the whole array")
arr = np.ones((3, 3))
arr *= 3
print(arr)
print("The operations can be any operations such as +, -, *, /, %, //")

# creating a numpy array from a list -> arr = np.array([1, 2, 3])
# creating a numpy zeros array -> zeros_arr = np.zeros((3, 3))

# optimizations to consider
# use integer 8 [-128, 127] rather than float 32 [-3.40282347E+38 to -1.17549435E-38] to save memory
# (8 bits compared to 32 bits)
# np.zeros((3, 3), dtype=np.int8) # dtype means data type
# list of dtypes: [np.bool_, np.int8, np.int16, np.int32, np.int64, np.float16, np.float64]

class Connect4:
    # I am charlotte
    # hi brian!!!
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
        self.policy_shape = (
        "your policy shape (length)",)  # MUST HAVE or else I won't be able to define the neural network
        # just know that the illegal moves are removed and the policy which is a probability distribution
        # is re normalized

    def get_current_player(self):
        # returns the current player
        pass

    def get_legal_actions(self):
        # returns the all possible legal actions in a list [action1, action2, ..., actionN] given self.board
        # Note that this action will be passed into do_action() and do_action_MCTS
        # MAKE SURE THERE are no duplicates (pretty self explanatory)
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

        # MAKE SURE THERE ARE NO DUPLICATES because it is going to increase the tree complexity thus slowing things down
        # it is also going to give a wrong policy and weird errors might occur
        # for checking use
        # assert len(legal_actions) == len(set(legal_action))
        # set cannot have any duplicates and thus removed so if the lengths are different the there is a problem
        pass

    def do_action(self, action):
        # places the move onto the board
        pass

    @staticmethod
    # @njit(cache=True)
    def do_action_MCTS(board, action, current_player):
        # this is for the monte carlo tree search's
        pass

    def get_input_state(self):
        # gets the numpy array for the neural network
        # for now just return the board as a numpy array
        # Brian will probably implement this later for specific neural networks
        # RWKV can just take in the board without problem
        # the original alphazero's network required the past boards
        # Uses in the root node of MCTS
        pass

    @staticmethod
    # @njit(cache=True)
    def get_input_state_MCTS(board):
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
        # Used to check if MCTS has reached a terminal node
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
