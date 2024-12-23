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

    def get_legal_actions(self): #-> list[tuple[int, int], ...] or np.array:
        """
        self.total_action is a set
        :return: a list of actions [action0, action1]
        """
        return list(self.total_actions - set(self.action_history)) # Note: O(n)


    @staticmethod
    # @njit(cache=True)
    def get_legal_actions_policy_MCTS(board: np.array, policy: np.array, shuffle=False):
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
    def _check_row_MCT(row):  # A function for checking if there's a win on each row.
        if row[0] != 0 and row[0] == row[1] == row[2]:
            return True  # If all 3 items on a row equal, someone has won. Yippee!
        return False  # If not then sadly not :(


    def check_win(self): #Checking if someone has won in rows
        for row in self.board:
            if self._check_row(row) is True: #Pulling out func from LALA-land
                "                    ^ Jessy, just letting you know that 'is True' is not necessary for python"
                "It's equally valid to do 'if self._check_row(row):' - Brian"
                return self.current_player #Someone has won! :D

        for column_index in range(3): #Checking if someone has won in columns
            column = self.board[:, column_index]
            if self._check_row(column) is True: #Pulling out func from LALA-land pt2
                return self.current_player #Someone has won, yippee :3

        diag1 = np.diag(self.board) #Checking if someone has won in diagonals
        if self._check_row(diag1) is True: #Pulling out func from dark magic
            return self.current_player #Someone has won :P

        diag2 = np.diag(np.fliplr(self.board))
        if self._check_row(diag2) is True: #Pulling func from Narnia
            return self.current_player #Someone has won :)

        flattened_board = self.board.reshape((9,)) #Reshape board into 9 items, this is to check for draw
        for value in flattened_board:
            if value == 0: #Still empty spaces on the board, stop checking!!
                break
        else:
            return 0 # return a draw if break never happens - Brian

        return -2 #Game continues :3

    @staticmethod
    # @njit(cache=True)
    def check_win_MCTS(board, last_action, current_player):
        for row in board:
            if TicTacToe._check_row_MCT(row):
                return current_player

        for column_index in range(3):
            column = board[:, column_index]
            if TicTacToe._check_row_MCT(column):
                return current_player

        diag1 = np.diag(board)
        if TicTacToe._check_row_MCT(diag1):
            return current_player

        diag2 = np.diag(np.fliplr(board))
        if TicTacToe._check_row_MCT(diag2):
            return current_player

        flattened_board = board.reshape((9,))
        for value in flattened_board:
            if value == 0:
                break
        else:
            return 0

        return -2

    @staticmethod
    @njit(cache=True)
    def get_winning_actions_MCTS(board, current_player, fast_check=False):
        # Brian will be looking very closely at this code when u implement this
        # Recommend to use check_win_MCTS unless there is a more efficient way
        # making sure that this doesn't slow this MCTS to a halt
        # if your game in every case only has 1 winning move you don't have to use fast_check param
        # please do not remove the fast_check parameter
        # check the gomoku example for more info
        pass

    def compute_policy_improvement(self, statistics):
        # given statistic=[[action, probability], ...] compute the new policy which should be of shape=self.policy_shape
        # example for tic tac toe statistics=[[[0, 0], 0.1], [[1, 0], 0.2], ...]
        # return [0.1, 0.2, ...]
        # this should map the action and probability to a probability distribution
        new_policy = np.zeros(self.policy_shape)
        for (x, y), prob in statistics:
            new_policy[x + 3 * y] = prob

        return new_policy


if __name__ == "__main__":
    # test your code here
    game = TicTacToe()
    game.do_action((0, 0))
    game.do_action((1, 0))
    game.do_action((2, 0))
    print(game.board)
    # game.do_action((0, 0))

    print(game.check_win())
    # print(game.get_legal_actions())
    # print(len(game.get_legal_actions()))
    # game.check_win()








    # todo Jump Ian Patrick Tang