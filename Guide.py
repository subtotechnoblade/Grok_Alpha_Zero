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

# This is the default model build config and will be passed to Gomoku_Build_Model_Time_Parallel.py
build_config = {"num_resnet_layers": 1,  # example: you can define your own variables and construct the network with these vars

                "use_stable_max": True,  # use stablemax, which will also use stablemax crossentropy
                "use_grok_fast": True,  # from grokfast paper
                "use_orthograd": True,  # from grokking at the edge of numerica stability
                "grok_lambda": 4.5,  # This is for grok fast, won't be used if model is Grok_Fast_EMA_Model
          }

train_config = {
    "total_generations": 100,  # Total number of generations, the training can be stopped and resume at any moment
    # a generation is defined by a round of self play, padding the dataset, model training, converting to onnx

    # Self Play variables
    "games_per_generation": 100,  # number of self play games until we re train the network
    "max_actions": 9,  # maximum actions allowed in a game
    "num_explore_actions_first": 2,  # A good rule of thumb is how long the opening should be for player -1
    "num_explore_actions_second": 0,  # Since player 1 is always at a disadvantage, we explore less and attempt to play better moves

    "use_gpu": True,  # Change this to false to use CPU for self play and inference
    "use_tensorrt": True,  # Assuming use_gpu is True, uses TensorrtExecutionProvider
    # change this to False to use CUDAExecutionProvider
    "use_inference_server": False,  # if an extremely large model is used, because of memory constraints, set this to True
    "max_cache_depth": 2,  # maximum depth in the search of the neural networks outputs we should cache
    "num_workers": 4,  # Number of multiprocessing workers used to self play

    # MCTS variables
    "MCTS_iteration_limit": 128,  # The number of iterations MCTS runs for. Should be 2 to 10x the number of starting legal moves
    # True defaults to iteration_limit = 3 * len(starting legal actions)
    "MCTS_time_limit": None,  # Not recommended to use for training  Set to True for a default of 30 seconds per move
    "use_njit": None,  # None will automatically infer what is supposed to be use for windows/linux

    "use_gumbel": True,  # use gumbel according to https://openreview.net/pdf?id=bERaNdoegnO
    # These params will only be used when use_gumbel is set to True
    "m": 16,  # Number of actions sampled in the first stage of sequential halving
    "c_visit": 50.0,
    "c_scale": 0.1,

    # These params will be used when use_gumbel is set to False
    "c_puct_init": 1.25,  # (shouldn't change) Exploration constant lower -> exploitation, higher -> exploration
    "dirichlet_alpha": 1.11,  # should be around (10 / average moves per game) this case is (10 / 9)

    "opening_actions": [],  # starting first move in the format [[action1, prob0], [action1, prob1], ...],
    # if prob doesn't add up to 1, then the remaining prob is for the MCTS move

    "num_previous_generations": 3,  # The previous generation's data that will be used in training
    "train_percent": 1.0,  # The percent used for training after the test set is taken
    "train_decay": 0.75,  # The decay rate for previous generations of data previous_train_percent = current_train_percent * train_decay
    "test_percent": 0.1,  # The percent of a dataset that will be used for validation
    "test_decay": 0.75,  # The decay rate for previous generations of data previous_test_percent = current_test_percent * test_decay

    "mixed_precision": None,  # None for no mixed precision, mixed_float16
    "train_batch_size": 8,  # The number of samples in a batch for training in parallel
    "test_batch_size": None,  # If none, then train_batch_size will be used for the test batch size
    "gradient_accumulation_steps": None,
    "learning_rate": 1e-3,  # Depending on how many layers you use. Recommended to be between 1e-3 to 5e-4
    "decay_lr_after": 20,  # When the n generations pass,... learning rate will be decreased by lr_decay
    "lr_decay": 0.5,  # multiplies this to learning rate every decay_lr_after
    "beta_1": 0.9,  # DO NOT TOUCH unless you know what you are doing
    "beta_2": 0.989,  # DO NOT TOUCH. This determines whether it groks or not. Hovers between 0.985 to 0.995
    "optimizer": "Nadam",  # optimizer options are ["Adam", "AdamW", "Nadam"]
    "train_epochs": 5,  # The number of epochs for training a generation's network
}
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
        self.next_player = -1
        # The DEFINITION of next_player is the player that is going to play but hasn't put their move on the board
        # The DEFINITION of current_player is the player that has just played and their move is on the board
        # to swap the current the player -> next_player *= -1 (very handy)
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
        self.policy_shape = ("shape",)  # MUST HAVE or else I won't be able to define the neural network
        # just know that the illegal moves are removed in get_legal_actions_policy_MCTS() and the policy which is a probability distribution
        # is re normalized

    def get_next_player(self) -> int:
        # returns the next player
        pass

    def input_action(self):
        # returns an action using input()
        # for tictactoe it is "return int(input()), int(input())"
        pass

    def get_legal_actions(self) -> np.array:
        # returns the all possible legal actions in a numpy array [action1, action2, ..., actionN] given self.board
        # Note that this action will be passed into do_action() and do_action_MCTS
        # MAKE SURE THERE are no duplicates (pretty self explanatory)
        pass
    @staticmethod
    # @njit(cache=True)
    def get_legal_actions_MCTS(board: np.array, current_player: int, action_history: np.array):
        # this returns the legal moves, just do the same thing as get_legal_actions
        pass

    @staticmethod
    # @njit(cache=True)
    def get_legal_actions_policy_MCTS(board: np.array,
                                      current_player: int,
                                      action_history: np.array,
                                      policy: np.array,
                                      normalize=True,
                                      shuffle=False) -> (np.array, np.array):
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

        # example with randomization for tictactoe:
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

    def do_action(self, action) -> np.array:
        # places the move onto the board
        pass

    @staticmethod
    # @njit(cache=True)
    def do_action_MCTS(board, action, next_player) -> np.array:
        # this is for the monte carlo tree search's
        pass

    def get_input_state(self) -> np.array:
        # gets the numpy array for the neural network
        # for now just return the board as a numpy array
        # Brian will probably implement this later for specific neural networks
        # the original alphazero's network required the past boards
        # Uses in the root node of MCTS
        pass

    @staticmethod
    # @njit(cache=True)
    def get_input_state_MCTS(board: np.array, current_player: int, action_history: np.array) -> np.array:
        # Used for retrieving the state for any child nodes (not the root)
        # just return the board from the inputs because board is also an np.array
        return board

    def check_win(self) -> int:
        # returns the player who won (-1 or 1), returns 0 if a draw is applicable
        # return -2 if no player has won / the game hasn't ended

        # feel free to implement the logic in check_win_MCTS
        # and call it like Game.check_win_MCTS(self.board, self.action_history[-1], self.current_player)
        # of course "Game" should be changed to the class name, "Game" is just an example
        pass

    @staticmethod
    # @njit(cache=True)
    def check_win_MCTS(board, current_player, action_history) -> int:
        # Used to check if MCTS has reached a terminal node
        # return the player (-1 or 1) if that player won, 0 if draw, and -2 if game hasn't ended
        # use the previous actions if necessary
        pass

    def compute_policy_improvement(self, statistics):
        # given [[action, probability], ...] compute the new policy which should be of shape=self.policy_shape
        # example for tic tac toe statistics=[[[0, 0], 0.1], [[1, 0], 0.2], ...] as in [[action0, probability for action0], ...]
        # you should return a board with each probability assigned to each move
        # return [0.1, 0.2, ...]
        # note that the coordinate [0, 0] corresponds to index 0 in the flattened board
        # this should map the action and probability to a probability distribution
        pass
    @staticmethod
    #@njit(cache=True)
    def augment_sample(input_states, policies):
        # optional method to improve convergence
        # rotate the board and flip it using numpy and return those as a list along with the original
        # remember to rotate the flip the policy in the same way as board


        # Note the optimal rotations and flips for tictactoe, and gomoku is
        # [original arr, flipup(arr), fliplr(arr)]
        # and [np.rot_90(arr, k = 1) + flipup(rot_arr), fliplr(rot_arr)]
        # and [np.rot_90(arr, k = 2)
        # and [np.rot_90(arr, k = 3)

        # Note that this won't be the case for connect4, and ult_tictactoe

        # Know that boards will be multiple boards of shape (num_augmentations, num_moves, board shape)
        # Know that the policies will be multiple policies (num_augmentations, num_moves, policy shape)

        # board_0 is board at move 0, board_1 is boar at move 1 and so on
        # Know that the expected output will be [[original_board_0, ...], [augmented_board_1, ...], ... for num_augmentations]
        #                                           ^original game          ^augmented game 1         ^augmented game 2 and so on
        # this will be the same for the policies

        # The expected output shape for the board will be (num_augmentations, num_actions, *board_shape)
        # The expected output shape for the policy will be (num_augmentations, num_actions, policy_shape[0])


        # If that is too hard, inform Brian and he can implement for you
        # Or
        return np.expand_dims(input_states, 0), np.expand_dims(policies, 0) # just return [board], [policy] if you don't want to implement this
        # don't be lazy, this will help convergence very much


# example for gomoku
class Gomoku:
    def __init__(self, width=15, height=15):
        self.board = np.zeros((height, width),
                              dtype=np.int8)  # note the dtype. Because I'm only using -1, 0, 1 int8 is best
        # if the board takes too much memory, Brian is not going to be happy
        self.next_player = -1
        self.action_history = []
        self.policy_shape = (225,)

    def get_next_player(self):
        return self.next_player

    def input_action(self):
        while True:
            try:
                coords = np.array(list(map(int, input("Action:").split(" "))))
                if self.board[coords[1]][coords[0]] == 0:
                    return coords
                print("Illegal move")
            except:
                print("Invalid input")


    def get_legal_actions(self) -> np.array:
        return self.get_legal_actions_MCTS(self.board, -self.next_player, np.array(self.action_history))
    @staticmethod
    # @njit(cache=True)
    def get_legal_actions_MCTS(board: np.array, current_player:int , action_history: np.array):
        """
        np.argwhere returns the index where the input array is 1 or True, in this case it return the indexes in format [[y, x], ...]
        where the board is empty (board == 0). [:, ::-1] reverses the [[y, x], ...] -> [[x, y], ...]
        """
        return np.argwhere(board == 0)[:, ::-1]

    @staticmethod
    @njit(cache=True)
    def get_legal_actions_policy_MCTS(board: np.array, current_player: int, action_history: np.array, policy: np.array, shuffle=False) -> (np.array, np.array):
        flattened_board = board.reshape(-1)  # makes sure that the board is a vector
        policy = policy[flattened_board == 0]  # keep the probabilities where the board is not filled

        # [board == 0] creates a mask and when the mask element is True, the probability at that index is returned
        policy /= np.sum(policy)
        # normalize the policy back to a probability distribution

        # reverse order of each element from [y, x] -> [x, y]
        legal_actions = np.argwhere(board == 0)[:, ::-1]  # get the indexes where the board is not filled

        # note that policy should already be flattened

        if shuffle:  # feel free to not implement this
            shuffled_indexes = np.random.permutation(len(legal_actions))  # create random indexes
            legal_actions, policy = legal_actions[shuffled_indexes], policy[
                shuffled_indexes]  # index the arrays to shuffled them
        # ^ example
        # [1, 2, 3], [0.1, 0.75, 0.15]
        # [1, 2, 0]
        # [2, 3, 1], [0.8, 0.15, 0.1]
        return legal_actions, policy

    def do_action(self, action) -> None:
        x, y = action

        assert self.board[y][x] == 0  # make sure that it is not an illegal move

        self.board[y][x] = self.next_player  # put the move onto the board
        self.next_player *= -1  # change players to the next player to play

        self.action_history.append(action)

    @staticmethod
    @njit(cache=True)
    def do_action_MCTS(board: np.array, action: tuple, next_player: int) -> np.array:
        x, y = action
        board[y][x] = next_player
        return board


    def get_input_state(self) -> np.array:
        return self.get_input_state_MCTS(self.board, -self.next_player, np.array(self.action_history))

    @staticmethod
    @njit(cache=True)
    def get_input_state_MCTS(board: np.array, current_player: int, action_history: np.array) -> np.array:
        return board

    def check_win(self, ) -> int:
        """
        # Note that this method is slow as it uses python for loops
        # recommend to use the check_win_MCTS with the board as njit
        # compiles and vectorizes the for loops
        :return: The winning player (-1, 1) a draw 1, or no winner -1
        """

        # use -self.current_player because in do_action we change to the next player but here we are checking
        # if the player that just played won so thus the inversion
        return self.check_win_MCTS(self.board, -self.next_player, np.array(self.action_history, dtype=np.int32),)

    @staticmethod
    @njit(cache=True, fastmath=True)
    def check_win_MCTS(board: np.array, current_player: int, action_history: np.array) -> int:
        """
        :return: The winning player (-1, 1) a draw 1, or no winner -1
        """

        current_x, current_y = action_history[-1]

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
        # if np.sum(np.abs(board.flatten())) == 15 * 15:
        #     return 0
        # ^ ostrich algorithm moment
        # remember that draw is very unlikely, but possible, just improbably

        # if there is no winner, and it is not a draw
        return -2

    def compute_policy_improvement(self, statistics):
        new_policy = np.zeros_like(self.board, dtype=np.float32)
        for (x, y), prob in statistics:
            new_policy[y][x] = prob
        return new_policy.reshape(-1)

    @staticmethod
    @njit(cache=True)
    def augment_sample_fn(states: np.array, policies: np.array):
        policies = policies.reshape((-1, 15, 15))# we need
        # to reshape this because we can only rotate a matrix, not a vector

        augmented_states = []
        augmented_policies = []

        for action_id in range(states.shape[0]):
            state = states[action_id]
            # augmented_board = np.zeros((8, board_shape))
            augmented_state = [state, np.flipud(state), np.fliplr(state)]


            policy = policies[action_id]
            augmented_policy = [policy, np.flipud(policy), np.fliplr(policy)]


            for k in range(1, 4):
                rot_state = np.rot90(state, k)
                augmented_state.append(rot_state)

                rot_policy = np.rot90(policy, k)
                augmented_policy.append(rot_policy)

                if k == 1:
                    augmented_state.append(np.flipud(rot_state))
                    augmented_state.append(np.fliplr(rot_state))

                    augmented_policy.append(np.flipud(rot_policy))
                    augmented_policy.append(np.fliplr(rot_policy))
            augmented_states.append(augmented_state)

            augmented_policies.append(augmented_policy)
        return augmented_states, augmented_policies

    def augment_sample(self, input_states, policies):
        # Note that values don't have to be augmented since they are the same regardless of how a board is rotated
        augmented_boards, augmented_policies = self.augment_sample_fn(input_states, policies)
        return np.array(augmented_boards, dtype=input_states[0].dtype).transpose([1, 0, 2, 3]), np.array(augmented_policies, dtype=np.float32).reshape((-1, 8, 225)).transpose([1, 0, 2])



if __name__ == "__main__":
    # example usage
    # game = Gomoku()
    # game.do_action((7, 7))
    # print(game.get_state())

    from Game_Tester import Game_Tester

    tester = Game_Tester(Gomoku)
    tester.test()

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