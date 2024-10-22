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
print("Starting Board")
print(board)
print("--"*10)
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
print(board)
print("draw")












# todo Jump Ian Patrick Tang



