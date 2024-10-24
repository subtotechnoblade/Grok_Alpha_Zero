import numpy as np
# this is going to be the format in which all game classes must 5 to

class Game:
    def __init__(self):
        # define your board as a numpy array
        # current player as an int, first player will be -1 second player is 1
        # to swap current the player -> current_player *= -1 (very handy)
        # move history as a list
        pass

    def get_current_player(self):
        # returns the current player
        pass

    def do_action(self, action):
        # places the move onto the board
        pass

    def get_state(self):
        # gets the numpy array for the neual network
        # for now just return the board
        # Brian will probably implement this later
        pass

    def check_win(self, board):
        # returns the player who won (-1 or 1), returns 0 if a draw is applicable
        # return -2 if no player has won / the game hasn't ended
        pass

# example
class Gomoku:
    def __init__(self):
        self.board = np.zeros((15, 15), dtype=np.int8)
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

    def get_state(self):
        return self.board

    def check_won(self, input_board: np.array, move: tuple) -> int:
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

if __name__ == "__main__":
    # example usage
    game = Gomoku()
    game.do_action((7, 7))
    print(game.get_state())
