import numpy as np
from numba import njit

# This is for building the model
build_config = {"embed_size": 128, # this is the vector for RWKV
          "num_heads": 2, # this must be a factor of embed_size or else an error will be raised
          "token_shift_hidden_dim": 32, # this is in the RWKV paper
          "hidden_size": None, # this uses the default 3.5 * embed size
          "num_layers": 3, # This is the total amount of RWKV layers in the model that are used
          }
class Gomoku:
    def __init__(self, width=15, height=15):
        self.board = np.zeros((height, width),
                              dtype=np.int8)  # note the dtype. Because I'm only using -1, 0, 1 int8 is best
        # if the board takes too much memory, Brian is not going to be happy
        self.current_player = -1
        self.action_history = []
        self.policy_shape = (225,)

    def get_current_player(self):
        return self.current_player

    def input_action(self):
        # try:
        #     coords = list(map(int, input().split(" ")))
        return np.array([1, 2, 3, 4])

    def get_legal_actions(self) -> np.array:
        # self.board == 0 creates a True and False board array, i.e., the empty places are True
        # np.argwhere of the mask returns the index where the mask is True, i.e. the indexes of the empty places are returned
        # ths returns [[1], [2], [3], ...] (shape=(-1, 1)) as an example but this is not what we want
        # (a -1 dimension means a N dimension meaning any length so it could mean (1, 1) or (234, 1))
        # we want [1, 2, 3, ...] (-1,) and thus reshape(-1)
        # [:, ::-1] reverses the order of the elements because argwhere returns [[y0, x0], ...] thus becomes [[x1, y0], ...]
        return np.argwhere(self.board == 0)[:, ::-1]
    @staticmethod
    @njit(cache=True)
    def get_legal_actions_MCTS(board):
        # same as the method above
        return np.argwhere(board == 0)[:, ::-1]

    @staticmethod
    @njit(cache=True)
    def get_legal_actions_policy_MCTS(board: np.array, policy: np.array, shuffle=False) -> (np.array, np.array):
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

        self.board[y][x] = self.current_player  # put the move onto the board
        self.current_player *= -1  # change players to the next player to play

        self.action_history.append(action)

    @staticmethod
    @njit(cache=True)
    def do_action_MCTS(board: np.array, action: tuple, current_player: int) -> np.array:
        x, y = action
        board[y][x] = current_player
        return board

    def get_state(self) -> np.array:
        return self.board

    @staticmethod
    @njit(cache=True)
    def get_state_MCTS(board: np.array) -> np.array:
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
        return self.check_win_MCTS(self.board, tuple(self.action_history[-1]), -self.current_player)

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
        # if np.sum(np.abs(board.flatten())) == 15 * 15:
        #     return 0
        # ^ ostrich algorithm moment
        # remember that draw is very unlikely, but possible, just improbably

        # if there is no winner, and it is not a draw
        return -2

    def compute_policy_improvement(self, statistics):
        new_policy = np.zeros_like(self.board)
        for (x, y), prob in statistics:
            new_policy[y][x] = prob
        return new_policy.reshape(-1)

    @staticmethod
    @njit(cache=True)
    def augment_array(arr):
        augmented_arrs = [arr, np.flipud(arr), np.fliplr(arr)]
        for k in range(1, 4):
            rot_arr = np.rot90(arr, k)
            augmented_arrs.append(rot_arr)
            if k == 1:
                augmented_arrs.append(np.flipud(rot_arr))
                augmented_arrs.append(np.fliplr(rot_arr))
        return augmented_arrs

    def augment_sample(self, board, policy):
        augmented_boards = self.augment_array(board)

        augmented_policies = []
        for augmented_policy in self.augment_array(policy.reshape((15, 15))): # we need
            # to reshape this because we can only rotate a matrix, not a vector
            augmented_policies.append(augmented_policy.reshape((-1,)))

        return augmented_boards, augmented_policies