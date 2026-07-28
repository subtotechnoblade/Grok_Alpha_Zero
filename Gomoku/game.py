import numpy as np
from numba import njit


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
        return self.get_legal_actions_MCTS(self.board, -self.next_player, np.array(self.action_history, dtype=np.uint8))
    @staticmethod
    @njit(cache=True)
    def get_legal_actions_MCTS(board: np.array, current_player:int , action_history: np.array):
        """
        np.argwhere returns the index where the input array is 1 or True, in this case it return the indexes in format [[y, x], ...]
        where the board is empty (board == 0). [:, ::-1] reverses the [[y, x], ...] -> [[x, y], ...]
        """
        return np.argwhere(board == 0)[:, ::-1].astype(np.uint8)

    @staticmethod
    @njit(cache=True)
    def get_legal_actions_policy_MCTS(board: np.array,
                                      current_player: int,
                                      action_history: np.array,
                                      policy: np.array,
                                      normalize=True,
                                      shuffle=False) -> (np.array, np.array):
        flattened_board = board.reshape(-1)  # makes sure that the board is a vector
        policy = policy[flattened_board == 0]  # keep the probabilities where the board is not filled

        # [board == 0] creates a mask and when the mask element is True, the probability at that index is returned
        if normalize:
            policy /= np.sum(policy)
        # normalize the policy back to a probability distribution

        # reverse order of each element from [y, x] -> [x, y]
        legal_actions = np.argwhere(board == 0)[:, ::-1].astype(np.uint8)  # get the indexes where the board is not filled

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

        self.action_history.append(np.array(action, dtype=np.uint8))

    @staticmethod
    @njit(cache=True)
    def do_action_MCTS(board: np.array, action: tuple, next_player: int) -> np.array:
        x, y = action
        board[y][x] = next_player
        return board


    def get_input_state(self) -> np.array:
        return self.get_input_state_MCTS(self.board, -self.next_player, np.array(self.action_history, dtype=np.uint8))

    @staticmethod
    @njit(cache=True)
    def get_input_state_MCTS(board: np.array, current_player: int, action_history: np.array) -> np.array:
        next_player_plane = np.ones_like(board) * -current_player
        return np.stack((next_player_plane, board), -1)


    def check_win(self) -> int:
        """
        # Note that this method is slow as it uses python for loops
        # recommend to use the check_win_MCTS with the board as njit
        # compiles and vectorizes the for loops
        :return: The winning player (-1, 1) a draw 1, or no winner -1
        """

        # use -self.current_player because in do_action we change to the next player but here we are checking
        # if the player that just played won so thus the inversion
        return self.check_win_MCTS(self.board, -self.next_player, np.array(self.action_history, dtype=np.uint8),)

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
        # remember that draw is very unlikely, but possible, just improbable

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
        return np.array(augmented_boards, dtype=self.board.dtype).transpose([1, 0, 2, 3, 4]), np.array(augmented_policies, dtype=np.float32).reshape((-1, 8, 225)).transpose([1, 0, 2])

if __name__ == "__main__":
    from Game_Tester import Game_Tester
    import time

    tester = Game_Tester(Gomoku)
    tester.test()

    # game = Gomoku()
    # game.do_action((7, 7))
    # print(game.action_history)
    # print(np.array(game.action_history).dtype)

    # game = Gomoku()
    # game.do_action((7, 7))
    # print(game.get_input_state().shape)
    # print(game.get_legal_actions())
    # boards = []
    # game.do_action((0, 0))
    # for _ in range(1):
    #     boards.append(game.board.copy())
    #
    # policies = np.ones((1, 225))
    # for i in range(10):
    #     augmented_boards, augmented_policies = game.augment_sample(np.array(boards), policies)
    #
    # s = time.time()
    # for i in range(100):
    #     augmented_boards, augmented_policies = game.augment_sample(np.array(boards), policies)
    # ttime = (time.time() - s)
    # print(ttime)
    # print(ttime / 100)

    # print(augmented_boards.shape)
    # print(augmented_policies.shape)