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

build_config = {"embed_size": 32, # this is the vector for RWKV
          "num_heads": 1, # this must be a factor of embed_size or else an error will be raised
          "token_shift_hidden_dim": 32, # this is in the RWKV paper
          "hidden_size": None, # this uses the default 3.5 * embed size
          "num_layers": 3, # This is the total amount of RWKV layers in the model that are used
          }


class TicTacToe:
    def __init__(self):
        self.current_player = -1
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.action_history = []
        # doesn't return anything

        self.policy_shape = (9,)

        self.total_actions = set() # precompute all possible actions at the start
        for y in range(3):
            for x in range(3):
                if self.board[y][x] == 0:
                    self.total_actions.add((x, y))
        # todo make move_history and add the played move everytime do_action is called

    def get_current_player(self):
        return self.current_player

    def input_action(self):
        coords = input().split(" ")
        coords[0], coords[1] = int(coords[0]), int(coords[1])

        if coords[0] < 0 or coords[0] > 2 or coords[1] < 0 or coords[1] > 2:
            raise ValueError ("Illegal move given")
        if self.board[coords[1]][coords[0]] != 0:
            raise ValueError ("Illegal move given")
        return np.array(coords)


    def get_legal_actions(self): #-> list[tuple[int, int], ...] or np.array:
        """
        self.total_action is a set
        :return: a list of actions [action0, action1]
        """
        return np.array(list(self.total_actions - set([(x, y) for x, y in self.action_history]))) # Note: O(n)


    @staticmethod
    # @njit(cache=True)
    def get_legal_actions_policy_MCTS(board: np.array, policy: np.array, shuffle: bool=False):
        legal_actions = []
        legal_policy = []
        for i, value in enumerate(board.reshape(-1)):
            if value == 0:
                legal_actions.append(np.array([i%3, i//3]))
                legal_policy.append(policy[i])
        legal_actions, legal_policy = np.array(legal_actions), np.array(legal_policy)

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
    # @njit(cache=True)
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

    @staticmethod
    @njit(cache=True)
    def _check_row_MCT(row):  # A function for checking if there's a win on each row.
        # If all 3 items on a row equal, someone has won. Yippee!
        # If not then sadly not :(
        # Brian has simplified it, speed is probably the same
        return row[0] != 0 and row[0] == row[1] == row[2]



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
    def check_win_MCTS(board, last_action, current_player):
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
        new_policy = np.zeros(self.policy_shape)
        for (x, y), prob in statistics:
            new_policy[x + 3 * y] = prob

        return new_policy

    @staticmethod
    #@njit(cache=True)
    def augment_array(arr):
        arr_augmentation = [arr, np.flipud(arr), np.fliplr(arr)]
        for k in range(1, 4):
            rotated_arr = np.rot90(arr, k)
            arr_augmentation.append(rotated_arr)
            if k == 1:
                arr_augmentation.append(np.fliplr(rotated_arr))
                arr_augmentation.append(np.flipud(rotated_arr))
        return arr_augmentation


    def augment_sample(self, board, policy):
        augmented_boards = self.augment_array(board)

        augmented_policies = []
        for augmented_policy in self.augment_array(policy.reshape(3, 3)):
            augmented_policies.append(augmented_policy.reshape(-1))

        return augmented_boards, augmented_policies


if __name__ == "__main__":
    # test your code here
    game = TicTacToe()
    game.do_action((0, 0))
    game.do_action((1, 1))
    game.do_action((2, 2))
    dummy_policy = np.random.uniform(low=0, high=1, size=(9,))
    print(game.get_legal_actions_policy_MCTS(game.board, dummy_policy))












    # todo Jump Ian Patrick Tang




    # rot90(), fliplr(), flipup()
    # we only need the original board + flipup + fliplr
    # and rot_board (k = 1) + flipup + fliplr
    # rot_board (k = 2)
    # rot_board (k = 3)