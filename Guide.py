import numpy as np
from numba import njit
# Disclaimer
# I WILL NOT CONFORM to open AI's gym,
# because there isn't a way to implement monte carlo tree search without game methods
# I'll conform to open AI's gym when I implement Muzero in the future as MCTS is no longer dependent on game methods
# Mu Zero is dependent on the representation and dynamics network to get the next state called the hidden state

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
        self.policy_shape = ("your policy shape (length)",) # MUST HAVE or else I won't be able to define the neural network
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
        #Used to check if MCTS has reached a terminal node
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


# example for gomoku
class Gomoku:
    def __init__(self):
        self.board = np.zeros((15, 15), dtype=np.int8) # note the dtype. Because I'm only using -1, 0, 1 int8 is best
        # if the board takes too much memory, Brian is not going to be happy
        self.current_player = -1
        self.action_history = []
        self.policy_shape = (225,)
    def get_current_player(self):
        return self.current_player

    def get_legal_actions(self):
        # self.board == 0 creates a True and False board array, i.e., the empty places are True
        # np.argwhere of the mask returns the index where the mask is True, i.e. the indexes of the empty places are returned
        # ths returns [[1], [2], [3], ...] (shape=(-1, 1)) as an example but this is not what we want
        # (a -1 dimension means a N dimension meaning any length so it could mean (1, 1) or (234, 1))
        # we want [1, 2, 3, ...] (-1,) and thus reshape(-1)

        return np.argwhere(self.board == 0).reshape(-1)
    @staticmethod
    @njit(cache=True)
    def get_legal_actions_policy_MCTS(board: np.array, policy: np.array, shuffle=False):
        board = board.reshape(-1) # makes sure that the board is a vector
        policy = policy[board == 0] # keep the probabilities where the board is not filled
        # [board == 0] creates a mask and when the mask element is True policy at that index is returned
        # normalize the policy back to a probability distribution
        policy /= np.sum(policy)

        legal_actions = np.argwhere(board[board == 0]).reshape(-1) # get the indexes where the board is not filled
        # note that policy should already be flattened
        if shuffle: # feel free to no implement this
            shuffled_indexes = np.random.permutation(len(legal_actions)) # create random indexes
            legal_actions, policy = legal_actions[shuffled_indexes], policy[shuffled_indexes] # index the arrays to shuffled them
        return legal_actions, policy



    def do_action(self, action):
        x, y = action

        assert self.board[y][x] == 0 # make sure that it is not an illegal move

        self.board[y][x] = self.current_player # put the move onto the board
        self.current_player *= -1 # change players to the next player to play

        self.action_history.append(action)
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
        # Note that this method is slow as it uses python for loops
        # recommend to use the check_win_MCTS with the board as njit
        # compiles and vectorizes the for loops
        :return: The winning player (-1, 1) a draw 1, or no winner -1
        """

        current_x, current_y = self.action_history[-1] # get the latest move
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
    @njit(cache=True, fastmath=True)
    def check_win_MCTS(board: np.array, last_action: tuple, current_player: int) -> int:
        """
        :return: The winning player (-1, 1) a draw 1, or no winner -1
        """

        current_x, current_y = last_action

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
    @staticmethod
    @njit(cache=True)
    def get_terminal_actions_MCTS(board, current_player, WIDTH=15, HEIGHT=15, fast_check=False):
        """
        :param board: The board
        :param current_player: Current player we want to check for
        :param WIDTH: board width
        :param HEIGHT: board height
        :param fast_check: only returns 1 winning move if True, should be False for MCTS,
        because we want multiple winning moves to determine a better policy
        :return:
        """
        legal_actions = np.argwhere(board == 0).reshape(-1)
        check_win_board = board.copy()
        terminal_actions = [] # includes winning and drawing actions
        terminal_mask = [] # a list of 0 and 1
        # where each index corresponds to a drawing action if 0, and a winning action if 1
        for legal_action in legal_actions:
            # Try every legal action anc check if the current player won
            # Very inefficient. There is a better implementation
            # for simplicity this will be the example
            x, y = legal_action % WIDTH, legal_action // HEIGHT
            check_win_board[y][x] = current_player
            result = Gomoku.check_win_MCTS(board, (x, y), current_player)
            if result != -2: # this limits the checks by a lot
                terminal_actions.append((x, y)) # in any case as long as the result != -2, we have a terminal action
                if result == current_player: # found a winning move
                    terminal_mask.append(1)
                    if fast_check:
                        break
                elif result == 0: # a drawing move
                    terminal_mask.append(0)

            check_win_board[y][x] = 0 # reset the board
        return terminal_actions, terminal_mask
if __name__ == "__main__":
    # example usage
    game = Gomoku()
    game.do_action((7, 7))
    print(game.get_state())

    """
    Brian's deep thinking, I have to write it somewhere
    proof that mu zero isn't dependent on game methods
    I thought that mu zero's MCTS needed to remove illegal moves in the tree well because how else is
    the tree able to get terminal moves?
    But that doesn't even need to happen, its even easier than alpha zero's tree because we allow "illegal" moves within the tree
    but when getting the improved policy from the MCTS, the "illegal" actions are just removed and the distribution is 
    normalized

    use quotations "illegal" because there is no way for us to know if it is or not, as the tree no longer uses a board
    but a hidden state which is similar to a compressed version of the board, which the representation and dynamic network
    generates
    """
