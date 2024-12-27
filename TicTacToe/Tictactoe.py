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

def foo(inputs):
    return inputs ** 2

# class
# class House:
#     def __init__(self, new_owner): # constructor
#         self.owner = "Jessica" # Jessica
#         self.owner = new_owner # Brian overwrites Jessica
#
# house = House("Brian") # this is calling constructor
# print(f"Ha Ha {house.owner} is the new owner")


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
        return list(self.total_actions - set(self.action_history)) # Note: O(n)


    @staticmethod
    # @njit(cache=True)
    def get_legal_actions_policy_MCTS(board: np.array, policy: np.array, shuffle: bool=False):
        # Note that board is a (3, 3) matrix
        # and policy is a (9,) vector

        # first task is to set every element in policy to 0 if the position on board is not empty (aka not 0)




        """
        THIS PART IS REALLY HARD FOR BEGINNERS, I RECOMMEND TO SKIP THIS PART UNTIL YOU ARE MORE CONFIDENT
        :param board: numpy array of the board
        :param policy: a numpy array of shape = self.policy shape defined in __init__, straight from the neural network's policy head
        :param shuffle: You might want to shuffle the policy and legal_actions because the last index is where the search starts
        if it is too hard you don't implement anything much will happen, its just that randomizing might improve convergence just by a bit
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

    def do_action(self, action): # must conform to this format - Brian
        x, y = action
        self.board[y][x] = self.current_player
        self.current_player = self.current_player * -1
        self.action_history.append(action)

        # Implement action_history

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
    print(game.get_legal_actions_policy_MCTS(game.baord, dummy_policy))












    # todo Jump Ian Patrick Tang




    # rot90(), fliplr(), flipup()
    # we only need the original board + flipup + fliplr
    # and rot_board (k = 1) + flipup + fliplr
    # rot_board (k = 2)
    # rot_board (k = 3)