import numpy as np

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
#  [0, 0, 0]]
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

outerboard = np.zeros((3, 3), dtype=np.int8)
def func(board, x, y, player):
    board[y][x] = player
    return board

# function vs method
# function

def foo(inputs):
    return inputs ** 2

# class
class House:
    def __init__(self, new_owner): # constructor
        self.owner = "Jessica" # Jessica
        self.owner = new_owner # Brian overwrites Jessica

house = House("Brian") # this is calling constructor
print(f"Ha Ha {house.owner} is the new owner")


class TicTacToe:
    def __init__(self):
        self.current_player = -1
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.action_history = []
        # doesn't return anything

        # todo make move_history and add the played move everytime do_action is called

    def do_action(self, action): # must conform to this format - Brian
        x, y = action
        self.board[y][x] = self.current_player
        self.current_player = self.current_player * -1

        # Implement action_history

    def check_win(self):
        # for getting a row use indexing board [0]
        # for getting a column use advanced indexing [:, 0] # where the 0 is the column number
        # for a diagonal (left to right) use np.trace()
        # for the right to left diagonal use np.fliplr (flips the board left to right) and np.trace to get the diagonal

        # then check the np.sum(row, dia, columns)
        # for checking if a draw check, Brian will write it for you
        if all(self.board != 0):
            return 0
        raise NotImplementedError("Implement check win with other methods")


game = TicTacToe()
game.do_action((1, 1))
game.do_action((0, 0))
print(game.board)
game.check_win()



# todo Jump Ian Patrick Tang

