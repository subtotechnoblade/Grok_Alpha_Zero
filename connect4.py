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
    # refer to Guide.py to implement each method, please be careful of what is used and returned
    # if you want pure speed recommend the decorator @njit for methods with for loops and numpy operations
    # contact Brian if you want more info
    def __init__(self):
        # todo create board
        pass

    def get_current_player(self):
        pass

    def get_legal_actions(self):
        pass

    @staticmethod
    # @njit(cache=True)
    def get_legal_actions_policy_MCTS(board: np.array, policy: np.array, shuffle=False):
        pass
    def do_action(self, action):
        # action is going to be a number from 0 to 6 representing the place to drop the piece
        pass

    @staticmethod
    # @njit(cache=True)
    def do_action_MCTS(board, action, current_player):
        # this is for the monte carlo tree search's
        pass

    def get_state(self):
        pass
    @staticmethod
    # @njit(cache=True)
    def get_state_MCTS(board):
        pass
    def check_win(self):
        pass

    @staticmethod
    # @njit(cache=True)
    def check_win_MCTS(board, last_action):
        #Used to check if MCTS has reached a terminal node
        pass

    @staticmethod
    # @njit(cache=True)
    def get_winning_actions_MCTS(board, current_player, fast_check=False):
        pass
