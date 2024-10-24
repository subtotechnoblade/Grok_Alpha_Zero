import numpy as np
#
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
#     board = [[0, 0, 0],
#              [0, 0, 0],
#              [0, 0, 0]]
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
        self.owner = "Brian" # Brian
        self.owner = new_owner # Jessy 2.0, over writes Brian

house = House("Jessy 2.0") # this is calling constructor
# print(house.owner)


class TicTacToe:
    def __init__(self):
        self.current_player = -1
        self.board = np.zeros((3, 3), dtype=np.int8)
        # doesn't return anything

    def place_move(self, x, y):
        self.board[y][x] = self.current_player
        self.current_player = self.current_player * -1

game = TicTacToe()
game.place_move(1, 1)
# game.place_move(0, 0)
# game.place_move(0, 0)
# print(game.board)



# todo Jump Ian Patrick Tang


class Container:
    def __init__(self, filter):
        self.filter = filter
        self.storage = []

    def add(self, item):
        if item in self.filter:
            self.storage.append(item)


container = Container([2, 3, 5, 7, 11])
container.add(2)
print(container.storage)
