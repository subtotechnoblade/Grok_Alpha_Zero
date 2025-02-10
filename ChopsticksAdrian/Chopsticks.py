import numpy as np

class Chopsticks:

    def __init__(self):
        self.board = np.array([1,2,3])



if __name__ == "__main__":
    from Game_Tester import Game_Tester
    game_tester = Game_Tester(Chopsticks,)
    game_tester.test()

