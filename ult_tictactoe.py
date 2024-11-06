import numpy as np
from numba import njit

class UltimateTicTacToe:
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
        #initialize board as a numpy array, first 9 entries are individual boards, and the 10th is the overall board
        self.board = np.zeros((10, 3, 3), dtype=np.int8)
        self.current_player = -1
        # action history as a list [action0, action1, ...]
        self.action_history = []
        self.policy_shape = (81,) 

    def get_current_player(self):
        return self.current_player
        
    '''
    def get_legal_actions(self):
        moves = []
        for sub_board_index in range (9):
            if (self.board[9][sub_board_index/3][sub_board_index%3] == 0):
                for x in range (3):
                    for y in range(3):
                        if(self.board[sub_board_index][x][y] == 0):
                            moves.append((sub_board_index,x,y))

        # ^ board[9] is going to give you an error, u probably meant self.board[9]
        # you should probably rename board to be called sub_board_index
        # don't confuse self.board and board
        return moves
        # returns an array of tuples (board number, x, y)
        # Note that this action will be passed into do_action() and do_action_MCTS
    '''
    def get_legal_moves(self):
        moves = []
        for sub_board_index, sub_board in enumerate(self.board):
            legal_action_indexes = np.argwhere(sub_board.reshape(-1) == 0).reshape(-1)
            print(legal_action_indexes)
            print(np.ones_like(legal_action_indexes) * sub_board_index, legal_action_indexes // 3, legal_action_indexes % 3)
            sub_board_legal_action_coords = zip(np.ones_like(legal_action_indexes) * sub_board_index, legal_action_indexes // 3, legal_action_indexes % 3)
            print(list(sub_board_legal_action_coords))
            moves += list(sub_board_legal_action_coords)
            raise ValueError("This is not an error just a debugging message, remove this once you understand")
        # for each sub board get the sub_board_index and sub_board from [(0, board0), (1, board1), ...]
        # the (0, board0) is generated from enumerate(self.board)
        # reshape the board into a vector, where y = element // 3 and x = element % 3
        # get the indexes that are 0 with sub_board == 0 aka empty reshaping the array to be of shape (-1,)
        # legal_actions for sub_board -> [0, 1, 2, 3, 4, 5, 6, 7, 8] assuming fully empty sub board

        # now each element gets floor dived by 3 gives the y index
        # giving for the y axis [0 0 0 1 1 1 2 2 2]

        # mod by 3 giving us the x (remainder is x index)
        # giving for the x axis [0 1 2 0 1 2 0 1 2]

        # np.ones_like(action_indexes) gives an array of length legal_action and multiplied by sub_board index
        # gives an array of repeating sub_board_indexes
        # putting it all together assuming the first sub board (sub_board_index 0)
        # zip([0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2])
        # -> [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 2, 0), (0, 2, 1), (0, 2, 2)]
        """
            ^note that this represents [(sub_board_index, y, x), ...] following the format of z, y, x
            feel free to change the order to get [(x, y, sub_board_index), ...]
            
        """
        # repeat for all suboards and have them all in a list
        return moves

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
        pass

    
    #Should also check if small square for current move is filled
    def do_action(self, action):
        z, y, x = action
        assert self.board[z][x][y] == 0 and self.board[9][z/3][z%3] == 0 #ensures move is legal
        '''
        Small note that z/3 does not give an integer it gives a float because python, z//3 is what you want
        floor division returns the result rounded down to the nearest integer, 5 // 2 = 2
        -Brian
        '''
        self.board[z][x][y] = self.current_player # put the move onto the board
        self.current_player *= -1 # change players to the next player to play

        self.action_history.append(action) #add action to action history
        pass
    
    @staticmethod
    # @njit(cache=True)
    def do_action_MCTS(board, action, current_player):
        # this is for the monte carlo tree search's
        pass

    def get_inputs_state(self):
        # gets the numpy array for the neural network
        # for now just return the board as a numpy array
        # Brian will probably implement this later for specific neural networks
        # RWKV can just take in the board without problem
        # the original alphazero's network required the past boards
        # Uses in the root node of MCTS
        pass

    @staticmethod
    # @njit(cache=True)
    def get_inputs_state_MCTS(board):
        # Used for retrieving the state for any child nodes (not the root)
        # just return the board from the inputs
        return board


    #this is an END OF GAME check, not a board section check
    def check_win(self):
        # returns the player who won (-1 or 1), returns 0 if a draw is applicable
        # return -2 if no player has won / the game hasn't ended
        board = np.array(self.board[9])
        "            ^ there is no need for np.array(self.board[9]) because that just creates a copy"
        numFilled = 0 # not needed anymore
        #rows
        sumRow = np.sum(board, axis=0)
        for val in sumRow:
            if (val == 3 or val == -3):
                return val/3
        #columns
        sumCol = np.sum(board, axis = 1)
        for val in sumCol:
            if (val == 3 or val == -3):
                return val/3
        
        #diagonals
        sumDia1 = np.trace(board)
        sumDia2 = np.trace(np.fliplr(board))
        if (sumDia1 == 3 or sumDia1 == -3): 
            return sumDia1/3
        if (sumDia2 == 3 or sumDia2 == -3): 
            return sumDia2/3
        
        #checks for tie
        if (board != 0).all():
            # board != 0 creates an array with the same shape as board as boolean values
            # each element of board gets replaced by element != 0 (boolean values)
            # all just checks if all of the values are True
            # giving something like [1, 0, 0] -> [True, False, False] -> False
            # i.e. if all of the elements are not 0, then True
            return 0
        return -2
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


if __name__ == "__main__":
    game = UltimateTicTacToe()
    print(game.get_legal_moves_Brian())
    # Test your functions and methods here
    # to call internal class methods use game.method()
    # to call a static method use UntimateTicTacToe.staticmethod()