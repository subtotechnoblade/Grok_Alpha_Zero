import numpy as np
from warnings import warn
import time
from Guide import Gomoku

class Game_Tester:
    def __init__(self, game_class, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.game_class = game_class
        self.game: Gomoku = self.game_class(*self.args, **self.kwargs)

    def reset(self):
        self.game = self.game_class(*self.args, **self.kwargs)

    def check_attributes(self):
        try:
            board = self.game.board
            print("Checking for a board attribute: Pass\n")
        except:
            print("Checking for a board attribute: Fail")
            print("The game class doesn't have a board attribute\n")
            return False

        if isinstance(self.game.board, np.ndarray):
            print("Checking if board is a numpy array: Pass\n")
        else:
            print("Checking if board is a numpy array: Fail")
            print("The board attribute in the game class must be a numpy array, and thus must be homogenous")
            print("If homogeneity is an issue, leave self.board as a flat array, and use reshape and slicing to construct the in other methods")
            print("Find Brian if this is a problem")
            return False


        try:
            current_player = self.game.current_player
            if current_player != -1:
                print("Starting current player isn't -1, recommend it to be -1")
                print("This is just a warning, there can be unforeseen bugs. It is probably fine")
            print("Checking for a current_player attribute: Pass\n")
        except:
            print("Checking for a board attribute: Fail")
            print("The game class doesn't have a current_player attribute")
            return False

        try:
            action_history = self.game.action_history
            if not isinstance(action_history, list):
                print("Action history must be a list")
                raise ValueError

            print("Checking for an action_history attribute: Pass\n")
        except:
            print("Checking for an action_history attribute: Failed\n")
            return False

        try:
            policy_shape = self.game.policy_shape
            if len(policy_shape) != 1:
                print("Make sure that policy shape is one number such as self.policy_shape = (9,)")
            print("Checking for a policy shape attribute: Pass\n")
        except:
            print("Checking for a policy shape attribute: Failed\n")
            print("Game class has no attribute policy shape, please define an attribute called")
            return False
        return True # meaning all tests have passed



    def check_legal_actions(self):
        try:
            legal_actions = self.game.get_legal_actions()
        except:
            print("Check legal actions: Fail\n")
            print("get_legal_actions isn't implemented check method name spelling if it is implemented")
            return False

        if len(legal_actions) == 0:
            print("Check legal actions: Fail\n")
            print("get_legal_actions gave no actions, must have actions at the start of the search")
            return False
        else:
            print("Checking if get_legal_actions returns actions: Pass\n")
            return True

    def check_legal_actions_policy_MCTS(self):
        class_legal_actions = self.game.get_legal_actions()

        dummy_policy = np.random.uniform(0, 1, size=self.game.policy_shape)

        try:
            MCTS_legal_actions, legal_policy = self.game.get_legal_actions_policy_MCTS(self.game.board, dummy_policy, shuffle=False)
        except AttributeError:
            print("Checking get_legal_actions_policy_MCTS: Fail")
            print("get_legal_actions_policy_MCTS isn't implemented")
            return False
        except TypeError:
            print("Checking get_legal_actions_policy_MCTS: Fail")
            print("get_legal_actions_policy_MCTS isn't implemented correctly, must return 2 numpy arrays")
            return False

        if not isinstance(MCTS_legal_actions, np.ndarray):
            print("Checking get_legal_actions_policy_MCTS: Fail")
            print("legal actions returned by get_legal_actions_policy_MCTS must be a numpy array")
            return False

        if not isinstance(legal_policy, np.ndarray):
            print("Checking get_legal_actions_policy_MCTS: Fail")
            print("The policy returned by get_legal_actions_policy_MCTS must be a numpy array")
            return False

        if len(class_legal_actions) != len(MCTS_legal_actions):
            print("Checking get_legal_actions_policy_MCTS: Fail")
            print("The list/array of legal actions returned by get_legal_actions and get_legal_actions_policy_MCTS must be the same length")
            return False

        for i in range(len(class_legal_actions)):
            if not np.array_equal(class_legal_actions[i], MCTS_legal_actions[i]):
                print("Checking get_legal_actions_policy_MCTS: Fail")
                print("Ordering for legal actions from get_legal_actions and get_legal_actions_policy_MCTS must be the same")
                return False

        shuffled_MCTS_legal_actions, shuffled_legal_policy = self.game.get_legal_actions_policy_MCTS(self.game.board, dummy_policy, shuffle=True)

        if (legal_policy != shuffled_legal_policy).any(): # check if arrays are of different order


            anti_shuffled_indexes = np.zeros_like(legal_policy, dtype=np.int32)
            for i, action in enumerate(MCTS_legal_actions):
                anti_shuffled_indexes[i] =  np.where((shuffled_MCTS_legal_actions == action).all(axis=1))[0][0]

            if (legal_policy != shuffled_legal_policy[anti_shuffled_indexes]).any():
                print("Checking shuffle in get_legal_actions_policy_MCTS: Fail")
                print("The shuffling for legal actions and policy are different")
                print("That is to say that you must shuffle both the legal actions and policy in the same way")
                print("The best way is to generate random indexes and append the policy and actions into 2 lists based on the randomly generated indexes\n")
                return False

            print("Checking shuffle in get_legal_actions_policy_MCTS: Pass\n")
        else:
            print("Shuffle isn't implementing skipping shuffle checks")
        print("Checking get_legal_actions_policy_MCTS: Pass\n")
        return True

    def check_do_action(self):
        legal_actions = self.game.get_legal_actions()
        try:
            for action in legal_actions:
                self.game.do_action(action)
        except AttributeError:
            print("Checking if do_action exists: Fail")
            print("do_action isn't implemented\n")
            return False
        self.reset()

        print("Checking if do_action exists: Pass\n")
        return True

    def check_do_action_MCTS(self):
        legal_actions = self.game.get_legal_actions()
        try:
            for action in legal_actions:
                board = self.game.do_action_MCTS(self.game.board.copy(), action, self.game.current_player)
        except AttributeError:
            print("Checking if do_action_MCTS exists: Fail")
            print("do_action_MCTS isn't implemented\n")
            return False

        self.reset()
        MCTS_board = self.game.board.copy()
        for action in legal_actions:
            MCTS_board = self.game.do_action_MCTS(MCTS_board, action, self.game.current_player)
            self.game.do_action(action) # this changes the player to the next, ^must happen before
            if not np.array_equal(MCTS_board, self.game.board):
                print("The returned board from do_action_MCTS doesn't match the one after do_action")
                print("Checking if do_action_MCTS returns the same thing as do_action: Fail\n")
                return False
        print("Checking if do_action_MCTS exists and returns the same board as do_action: Pass\n")
        self.reset() # reset the board for any future checks that changes the board
        return True

    def _check_action_history_homogenous(self):
        try:
            np.array(self.game.action_history)
            print("Checking if action history can be converted into a homogenous array: Pass\n")
            return True
        except:
            print(f"{self.game.action_history} could not be converted into a numpy array")
            print("Checking if action history can be converted into a homogenous array: Fail\n")
            return False

    def check_check_win(self):
        legal_actions = self.game.get_legal_actions()
        self.game.do_action(legal_actions[0])
        try:
            winner = self.game.check_win() # also counts as a warmup
        except AttributeError:
            print("Checking check_win: Fail")
            print("check_win not implemented")
            return False
        except:
            print("Checking check_win: Fail")
            print("check_win isn't callable as it gives an error")
            return False

        if winner != -2:
            print("Checking check_win: Fail")
            print("At the start of the game there cannot be a winner")
            return False
        self.reset()
        legal_actions = self.game.get_legal_actions()
        checks = 0
        total_check_win_time = 0
        winner = -2
        while winner == -2:
            if len(legal_actions) == 0 and winner == -2:
                print("Checking check_win: Fail")
                print("Board has been filled and check win still returned -2")
            chosen_random_index = np.random.choice(np.arange(len(legal_actions), dtype=np.int32), size=(1,))[0]
            self.game.do_action(legal_actions[chosen_random_index])
            t = time.time()
            winner = self.game.check_win()
            total_check_win_time += time.time() - t
            checks += 1
            legal_actions = self.game.get_legal_actions()


        print(f"Average time per check_win call: {total_check_win_time / checks} seconds averaged over 1 game of {checks} moves. Winner was: {winner}")
        if (total_check_win_time / checks) >= 1.0:
            print("Check_win implementation is inefficient and slow (time taken >= 1 second), optimizations may be possible")
        print("Check if the final board is c osdrrect (win or draw), since there is no way for this program to know that!")
        print(f"The winner should be: {winner}")
        print(self.game.board)
        print("Checking check_win: Pass\n")


        self._check_action_history_homogenous()

        self.reset()
        return True


    def test(self):
        test_skipped = 0
        if not self.check_attributes():
            print("Tests cannot continue unless all class attributes are created")
            return
        if not self.check_legal_actions():
            print("Tests cannot continue unless the tester can get the legal_actions from get_legal_actions")
            return

        if not self.check_legal_actions_policy_MCTS():
            print("Skipped checking legal_actions_policy_MCTS because it failed for reason above^")
            print("Tests will continue\n")
            test_skipped += 1

        if not self.check_do_action():
            return

        if not self.check_do_action_MCTS():
            print("Skipped checking do_action_MCTS because it failed for the reason above^")
            print("Tests will continue\n")
            test_skipped += 1

        if not self.check_check_win():
            return

        print("DISCLAIMER: currently these test are only testing around 50% of the functionality that the game class requires")
        print("I haven't written the tests for checking for wins/draws or getting terminal moves\n")

        if test_skipped == 0:
            print("All major and minor tests passed")
        else:
            print(f"Skipped {test_skipped} tests, scroll up to see")
            print("Aside from that all major tests passed")




if __name__ =="__main__":
    # Example usage
    from Guide import Gomoku
    # game_tester = Game_Tester(Gomoku, width=15, height=15)# if you have no game parameters, leave it blank
    from Tictactoe import TicTacToe
    game_tester = Game_Tester(TicTacToe,)
    game_tester.test()