import numpy as np

class Chopsticks:

    def __init__(self):
        self.board = np.array([], dtype=np.int8)
        self.current_player = -1 #p1 = -1, p2 = +1
        self.action_history = []
        self.policy_shape = (10,)

    def get_current_player(self):
        return self.current_player

    def input_action(self):
        #Takes "a b c" as input, returns [a b c]
        action = map(int, input("Action:").split(" "))
        return np.array(action)

    def get_legal_actions(self):
        LegalActions = [
            [0, ]
        ]




if __name__ == "__main__":
    from Game_Tester import Game_Tester
    game_tester = Game_Tester(Chopsticks,)
    game_tester.test()

