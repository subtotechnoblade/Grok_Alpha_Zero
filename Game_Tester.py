import numpy as np
import time


class Game_Tester:
    def __init__(self, game_class):
        self.game_class = game_class
        if not isinstance(game_class, type):
            raise TypeError("Please give the class, not an instance, example: Game_Tester(Game) not Game_Tester(Game())")
        try:
            self.game = self.game_class()
        except:
            print("You cannot specify parameters in the game's __init__ constructor")
            print("All parameters should be a variable in __init__, nothing should be passed in")
            raise TypeError("Class initiation failed")


    def reset(self):
        self.game = self.game_class()

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
            print("Checking for a current player attribute: Fail")
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
            print("Check get_legal_actions: Fail\n")
            print("get_legal_actions isn't implemented check method name spelling if it is implemented")
            return False
        if legal_actions is None:
            print("Check get_legal_actions: Fail\n")
            print("get_legal_actions returned None")
            return False

        if len(legal_actions) == 0:
            print("Check get_legal_actions: Fail\n")
            print("get_legal_actions gave no actions, must have actions at the start of the search")
            return False
        for action in legal_actions:
            count = np.count_nonzero([mask.all() for mask in legal_actions == action])
            if count > 1:
                print("Check get_legal_actions: Fail\n")
                print(f"There was a duplicate action: {action} found in {legal_actions}")
                return False
        print("Checking if get_legal_actions returns actions: Pass\n")
        return True

    def check_legal_actions_MCTS(self):
        try:
            legal_actions = self.game.get_legal_actions()
            MCTS_legal_actions = self.game.get_legal_actions_MCTS(self.game.board,
                                                                  self.game.get_current_player(),
                                                                  np.array(self.game.action_history))
        except:
            print("Check get_legal_actions_MCTS: Fail\n")
            print("get_legal_actions_MCTS isn't implemented, check method name if it is implemented")
            return False

        if len(legal_actions) != len(MCTS_legal_actions):
            print("Check get_legal_actions_MCTS: Fail\n")
            print("len of get_legal_actions can len of get_legal_actions MCTS are different")
            print("Check implementation")
            return False

        if not np.array_equal(legal_actions, MCTS_legal_actions):
            print("Check get_legal_actions_MCTS: Fail\n")
            print("get_legal_actions and get_legal_actions_MCTS give different results")
            return False

        print("Checking if get_legal_actions returns actions and validated against get_legal_actions: Pass\n")
        return True

    def check_legal_actions_policy_MCTS(self):
        class_legal_actions = self.game.get_legal_actions()

        dummy_policy = np.random.uniform(0, 1, size=self.game.policy_shape)

        try:
            MCTS_legal_actions, legal_policy = self.game.get_legal_actions_policy_MCTS(self.game.board,
                                                                                       self.game.get_current_player(),
                                                                                       np.array(self.game.action_history),
                                                                                       dummy_policy, shuffle=False)
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
            print("The list/array of legal actions returned by get_legal_actions and get_legal_actions_policy_MCTS must be the same length with the same elements")
            return False

        shuffled_MCTS_legal_actions, shuffled_legal_policy = self.game.get_legal_actions_policy_MCTS(self.game.board,
                                                                                                     self.game.get_current_player(),
                                                                                                     np.array(self.game.action_history),
                                                                                                     dummy_policy, shuffle=True)

        if (legal_policy != shuffled_legal_policy).any(): # check if arrays are of different order


            anti_shuffled_indexes = np.zeros_like(legal_policy, dtype=np.int32)
            for i, action in enumerate(MCTS_legal_actions):
                anti_shuffled_indexes[i] = np.where((shuffled_MCTS_legal_actions == action).all(axis=1))[0][0]
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

        if isinstance(self.game.action_history[0], list):
            print("Checking if do_action exists: Fail")
            print("Your action can't be a list")
            print("Recommend it to be a numpy array")
            return False
        elif isinstance(self.game.action_history[0], tuple):
            print("Recommend that your action be a numpy array, currently it is a tuple")
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
            print(f"At the start of the game there cannot be a winner returned: {winner}")
            return False

        try:
            self.game.check_win_MCTS(self.game.board, np.array(self.game.action_history), -self.game.get_current_player()) # also counts as a warmup
            MCTS_check_win_implemented = True
        except AttributeError:
            print("Checking check_win_MCTS: Fail")
            print("check_win_MCTS not implemented, skipped")
            MCTS_check_win_implemented = False

        except:
            print("Checking check_win_MCTS: Fail")
            print("check_win_MCTS isn't callable as it gives an error, skipped")
            MCTS_check_win_implemented = False


        self.reset()
        legal_actions = self.game.get_legal_actions()
        checks = 0
        total_check_win_time = 0
        total_check_win_MCTS_time = 0
        winner = -2
        current_player = -1
        while winner == -2:
            if len(legal_actions) == 0 and winner == -2:
                print("Checking check_win: Fail")
                print("Board has been filled and check win still returned -2")
                return  False
            chosen_random_index = np.random.choice(np.arange(len(legal_actions), dtype=np.int32), size=(1,))[0]
            action = legal_actions[chosen_random_index]
            self.game.do_action(action)

            s = time.time()
            winner = self.game.check_win()
            total_check_win_time += time.time() - s

            s = time.time()
            MCTS_winner = self.game.check_win_MCTS(self.game.board, np.array(self.game.action_history), -self.game.current_player)
            total_check_win_MCTS_time += time.time() - s

            if MCTS_check_win_implemented and winner != MCTS_winner:
                print("Checking check_win and check_win_MCTS: Fail")
                print(f"check_win: ({winner}) and check_win_MCTS: ({MCTS_winner}) doesn't match up")
                return False

            if winner  == -2:
                current_player *= -1
            checks += 1
            legal_actions = self.game.get_legal_actions()

        if abs(winner) == 1 and current_player != winner:
            print("If check_win found a winning board then you have to return the previous player")
            print("In do_action the player is changed to the next player to play, but the previous player already won")
            print("Just flip the current player to the previous player by multiplying by -1")
            print("Remember to do the same for check_win_MCTS or else an error will say that they don't match up!")
            return False

        print(f"Average time per check_win call: {total_check_win_time / checks} seconds averaged over 1 game of {checks} moves. Winner is: {winner}")
        if (total_check_win_time / checks) >= 1.0:
            print("Check_win implementation is inefficient and slow (time taken >= 1 second), optimizations may be possible\n")

        print(f"Average time per check_win_MCTS call: {total_check_win_MCTS_time / checks} seconds averaged over 1 game of {checks} moves. Winner is: {winner}")
        if (total_check_win_MCTS_time / checks) >= 1.0:
            print("check_win_MCTS implementation is inefficient and slow (time taken >= 1 second), optimizations may be possible\n")
        else:
            print()

        if total_check_win_MCTS_time > 0.0 and total_check_win_time > 0.0:
            if (total_check_win_MCTS_time / checks) / (total_check_win_time / checks) > 5:
                print("check_win is much faster than check_win_MCTS, since they are doing the same thing, check their implementation\n")

            if (total_check_win_time / checks) / (total_check_win_MCTS_time / checks) > 5:
                print("check_win_MCTS is much faster than check_win, perhaps use check_win_MCTS as the implementation for check_win?\n")

        print("Check if the final board is correct (win or draw), since there is no way for this program to know that!")
        print(f"Final action is {action}, and the winner should be: {winner}")
        print(self.game.board)
        print("Checking check_win: Pass\n")


        self._check_action_history_homogenous()

        self.reset()
        return True

    def check_compute_policy_improvement(self):
        dummy_stat = [(self.game.get_legal_actions()[0], 1.0), ]
        try:
            improved_policy = self.game.compute_policy_improvement(dummy_stat)
            if np.sum(improved_policy) != 1.0:
                print("Checking compute_policy_improvement: Fail")
                print("It isn't implemented correctly")
                return False
        except:
            print("Checking compute_policy_improvement: Fail")
            print("It is not implemented")

        print("Checking compute_policy_improvement: Pass\n")
        return True


    def check_augment_sample(self):
        legal_actions = self.game.get_legal_actions()
        self.game.do_action(legal_actions[np.random.randint(0, len(legal_actions), size=1)[0]])
        dummy_boards = np.array([self.game.board for _ in range(2)])
        dummy_policy = np.random.uniform(low=0, high=1.0, size=(2, *self.game.policy_shape))
        try:
            augmented_boards, augmented_policies = self.game.augment_sample(dummy_boards, dummy_policy)
        except:
            print("Checking augment sample: Fail")
            print("augment_sample isn't implemented")
            return False

        if not (isinstance(augmented_boards, np.ndarray) and isinstance(augmented_policies, np.ndarray)):
            print("Checking augment sample: Fail")
            print("augment_sample must return a np.array")


        if len(augmented_boards) == 0 or len(augmented_policies) == 0:
            print("Checking augment sample: Fail")
            print("augment_sample cannot return an empty list/array")
            return False
        elif len(augmented_boards) == 1 and len(augmented_policies) == 1:
            print("augment_sample isn't fully implemented")
            print("Recommend implementing this for better training")

        if augmented_boards.shape[1] != 2 or augmented_policies.shape[1] != 2:
            print("Checking augment sample: Fail")
            print("The timestep dimension should be the same and should be 2 (testing on 2 boards)")
            return False
        if augmented_boards.shape[2:] != self.game.board.shape:
            print("Checking augment sample: Fail")
            print("The dimensions after the first axis should be equal to board shape")

        if augmented_policies.shape[2:] != self.game.policy_shape:
            print("Checking augment sample: Fail")
            print("The dimensions after the first axis should be equal to policy shape")

        self.game = self.game_class()
        print(f"Augment sample duplicates each game by {augmented_boards.shape[0]} times!")
        print("Checking augment sample: Pass\n")
        return True


    def test(self):
        test_skipped = 0
        if not self.check_attributes():
            print("Tests cannot continue unless all class attributes are created")
            return
        if not self.check_legal_actions():
            print("Tests cannot continue unless the tester can get the legal_actions from get_legal_actions")
            return

        if not self.check_legal_actions_MCTS():
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

        if not self.check_compute_policy_improvement():
            return

        if not self.check_augment_sample():
            return

        if test_skipped == 0:
            print("All major and minor tests passed")
        else:
            print(f"Skipped {test_skipped} tests, scroll up to see")
            print("Aside from that all major tests passed")




if __name__ =="__main__":
    # Example usage
    # from Guide import Gomoku
    from Gomoku.Gomoku import Gomoku
    # game_tester = Game_Tester(Gomoku, width=15, height=15)# if you have no game parameters, leave it blank
    from TicTacToe.Tictactoe import TicTacToe
    game_tester = Game_Tester(TicTacToe)
    game_tester.test()