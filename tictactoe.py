import numpy as np

# make zeros array np.zeros(shape)
# make ones array np.ones(shape)
# make array from list np.array(list)

# human friendly board, not computer friendly
[["E", "E", "E"]]
[["E", "X", "O"]]
[["E", "X", "O"]]

[[0, 0, 0],
 [0, 0, 0],
 [0, 0, 0]]
# pieces with -1, 0, 1

board = np.zeros((3, 3), dtype=np.int8)
# numpy array board
# [[0 0 0]
#  [0 0 0]
#  [0 0 0]]
# print("Starting Board")
# print(board)
# print("--"*10)
# Next thing

board[0][0] = -1
board[1][1] = 1
board[0][1] = -1
board[0][2] = 1
board[2][0] = -1
board[1][0] = 1
board[1][2] = -1
board[2][2] = 1
board[2][1] = -1
# print(board)
# print("draw")

#todo functions: make a place move function that takes:
# the board
# coordinate x, y
# and the current player as -1 or 1
# places the move onto the board
# returns the board with the newly placed board


# Resources
# making a function
def my_func():
 return 100 # this can be anything, including numpy arrays

# calling a function
my_func()
x = my_func() # this will be 100
print(x)

# more examples
def add(a, b):
 return a + b

result = add(1, 1) # result should be 2
print(result)














# todo Jump Ian Patrick Tang



