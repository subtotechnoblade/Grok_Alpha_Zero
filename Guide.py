import numpy as np
from numba import njit
# Disclaimer
# I WILL NOT CONFORM to open Ai's gym,
# because there isn't a way to implement monte carlo tree search without game methods
# I'll conform to open AI's gym when I implement Muzero in the future as MCTS is no longer dependent on game methods


# this is going to be the format in which all game classes must conform to
# players are represented with -1 and 1 <- this rule cannot change
# game board must be a numpy array

class Game:
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
    # find Brian because there are a lot of restrictions
    Example

    @staticmethod
    @njit(cache=True) # For the first call, jit will be slower but after a warmup (compile time), it should be faster
    def do_action(board, x, y, current_player):
        board[y][x] = current_player
        return board
    """
    def __init__(self):
        # define your board as a numpy array
        # current player as an int, first player will be -1 second player is 1
        # to swap current the player -> current_player *= -1 (very handy)
        # move history as a list


        # must also define policy shape
        # for tictactoe the shape of (3, 3) is expected because there are 9 possible moves
        # for connect 4 the shape is (7,) because there are only 7 possible moves
        # for connect 5 it is (15, 15)
        # alternatively you can also request for a flattened policy and reshape it in parse_policy (not recommended)
        # in parse policy you will have to convert the policy into actions and the associated prob_prior
        self.policy_shape = (3, 3) # MUST HAVE or else something I wont be able to define the neural network, this is for tictactoe
        pass

    def get_current_player(self):
        # returns the current player
        pass

    def get_legal_actions(self):
        # returns the all possible legal actions in a list [action1, action2, ..., actionN] given self.board
        # Note that this action will be passed into do_action() and do_action_MCTS
        pass
    @staticmethod
    # @njit(cache=True)
    def get_legal_actions_MCTS(board,):
        # returns the all possible legal actions in a list [action1, action2, ..., actionN] given board
        # recommend to use numba njit to speed up MCTS
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
        # Brian will probably implement this later
        # Uses in the root node of MCTS
        pass

    @staticmethod
    # @njit(cache=True)
    def get_state_MCTS(board):
        # Used for retrieving the state for any child nodes (not the root)
        # just return the board from the inputs
        return board

    def check_win(self, board, last_action=None):
        # returns the player who won (-1 or 1), returns 0 if a draw is applicable
        # return -2 if no player has won / the game hasn't ended
        pass
    @staticmethod
    # @njit(cache=True)
    def check_win_MCTS(self, board, last_action):
        #Used to check if MCTS has reached a terminal node
        pass


# example for gomoku
class Gomoku:
    def __init__(self):
        self.board = np.zeros((15, 15), dtype=np.int8) # note the dtype. Because I'm only using -1, 0, 1 int8 is best
        # if the board takes too much memory, Brian is not going to be happy
        self.current_player = -1
        self.move_history = []
    def get_current_player(self):
        return self.current_player

    def do_action(self, action):
        x, y = action

        assert self.board[y][x] == 0 # make sure that it is not an illegal move

        self.board[y][x] = self.current_player # put the move onto the board
        self.current_player *= -1 # change players to the next player to play

        self.move_history.append(action)
    @staticmethod
    @njit(cache=True)
    def do_action_MCTS(board: np.array, action: tuple, current_player:int) -> np.array:
        x, y = action
        board[y][x] = current_player
        return board


    def get_state(self) -> np.array:
        return self.board
    @staticmethod
    @njit(cache=True)
    def get_state_MCTS(board: np.array) -> np.array:
        return board

    def check_win(self,) -> int:
        """
        :return: The winning player (-1, 1) a draw 1, or no winner -1
        """

        current_x, current_y = self.move_history[-1]
        player = self.current_player

        fives = 0
        for i in range(-5 + 1, 5):
            new_x = current_x + i
            if 0 <= new_x <= 15 - 1:
                if self.board[current_y][new_x] == player:
                    fives += 1
                    if fives == 5:
                        return player
                else:
                    fives = 0

        # vertical
        fives = 0
        for i in range(-5 + 1, 5):
            new_y = current_y + i
            if 0 <= new_y <= 15 - 1:
                if self.board[new_y][current_x] == player:
                    fives += 1
                    if fives == 5:
                        return player
                else:
                    fives = 0

        #  left to right diagonal
        fives = 0
        for i in range(-5 + 1, 5):
            new_x = current_x + i
            new_y = current_y + i
            if 0 <= new_x <= 15 - 1 and 0 <= new_y <= 15 - 1:
                if self.board[new_y][new_x] == player:
                    fives += 1
                    if fives == 5:
                        return player
                else:
                    fives = 0

        # right to left diagonal
        fives = 0
        for i in range(-5 + 1, 5):
            new_x = current_x - i
            new_y = current_y + i
            if 0 <= new_x <= 15 - 1 and 0 <= new_y <= 15 - 1:
                if self.board[new_y][new_x] == player:
                    fives += 1
                    if fives == 5:
                        return player
                else:
                    fives = 0
        # if sum([abs(x) for x in input_board.flat]) == 15 * 15:
        #     return 0
        # remember that draw is very unlikely, but possible

        # if there is no winner, and it is not a draw
        return -2
    @staticmethod
    @njit(cache=True)
    def check_win_MCTS(board: np.array, last_move: tuple, current_player: int) -> int:
        """
        :return: The winning player (-1, 1) a draw 1, or no winner -1
        """

        current_x, current_y = last_move

        fives = 0
        for i in range(-5 + 1, 5):
            new_x = current_x + i
            if 0 <= new_x <= 15 - 1:
                if board[current_y][new_x] == current_player:
                    fives += 1
                    if fives == 5:
                        return current_player
                else:
                    fives = 0

        # vertical
        fives = 0
        for i in range(-5 + 1, 5):
            new_y = current_y + i
            if 0 <= new_y <= 15 - 1:
                if board[new_y][current_x] == current_player:
                    fives += 1
                    if fives == 5:
                        return current_player
                else:
                    fives = 0

        #  left to right diagonal
        fives = 0
        for i in range(-5 + 1, 5):
            new_x = current_x + i
            new_y = current_y + i
            if 0 <= new_x <= 15 - 1 and 0 <= new_y <= 15 - 1:
                if board[new_y][new_x] == current_player:
                    fives += 1
                    if fives == 5:
                        return current_player
                else:
                    fives = 0

        # right to left diagonal
        fives = 0
        for i in range(-5 + 1, 5):
            new_x = current_x - i
            new_y = current_y + i
            if 0 <= new_x <= 15 - 1 and 0 <= new_y <= 15 - 1:
                if board[new_y][new_x] == current_player:
                    fives += 1
                    if fives == 5:
                        return current_player
                else:
                    fives = 0
        # if sum([abs(x) for x in input_board.flat]) == 15 * 15:
        #     return 0
        # remember that draw is very unlikely, but possible

        # if there is no winner, and it is not a draw
        return -2

if __name__ == "__main__":
    # example usage
    game = Gomoku()
    game.do_action((7, 7))
    print(game.get_state())
