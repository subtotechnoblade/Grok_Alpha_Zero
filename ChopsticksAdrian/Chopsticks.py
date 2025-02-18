import numpy as np

class Chopsticks:

    def __init__(self):
        self.board = np.array([[1, 1], [1, 1]], dtype=np.int8)
        self.current_player = -1 #p1 = -1, p2 = +1
        self.action_history = []
        self.policy_shape = (10,)

    def get_current_player(self):
        return self.current_player

    def input_action(self):
        #Takes "a b c" as input, returns [a b c]
        action = map(int, input("Action: ").split(" "))
        #Check if legal actions are in here
        return np.array(action)

    def get_legal_actions(self):
        # legal_actions = [
        #     [0, 0, 0],
        #     [0, 0, 1],
        #     [0, 1, 0],
        #     [0, 1, 1],
        #
        #     [1, 0, 1],
        #     [1, 0, 2]
        #
        #
        # ]

    @staticmethod
    # @njit(cache=True)
    def get_legal_actions_MCTS(board: np.array, current_player: int, action_history: np.array):

        #can't attack dead hand, can't attack with dead hand
        #for splits: can't reverse, can't equal five, can't equal zero
        #if given action breaks board,
        # own_hands = board[0]
        # opp_hands = board[1]
        hands = board.flatten()
        legal_actions = []
        #find where own hands != 0, find where opp hands != 0, write combos
        #attack
            #board.flatten, check if index
            #theorem/law: opponent cannot win on your move, you cannot win on your opponent's move
                #check own hand, no need to check opponent, essentially just check where you placed the move
        if hands[0] != 0:
            if hands[2] != 0:
                legal_actions += [0, 0, 0]
            if hands[3] != 0:
                legal_actions += [0, 0, 1]
        if hands[1] != 0:
            if hands[2] != 0:
                legal_actions += [0, 1, 0]
            if hands[3] != 0:
                legal_actions += [0, 1, 1]

        #split





        pass




if __name__ == "__main__":
    from Game_Tester import Game_Tester
    game_tester = Game_Tester(Chopsticks,)
    game_tester.test()

