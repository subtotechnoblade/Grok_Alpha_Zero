import numpy as np
from warnings import warn
from numba import njit
import numba as nb
from tqdm import tqdm
from Guide import Gomoku
import time
class Node:
    __slots__ = "child_id", "board", "action_history", "current_player", "children", "child_legal_actions", "child_visits", "child_values", "RNN_state", "child_prob_priors", "is_terminal",  "parent"
    def __init__(self,
                 child_id: int,
                 board: np.array,
                 action_history: np.array,
                 current_player: int,
                 child_legal_actions:list[tuple[int]] or list[int],
                 RNN_state:list or list[np.array, ...] or np.array,
                 child_prob_priors: np.array,
                 is_terminal=None,
                 parent=None):
        self.child_id = child_id
        self.board = board # board, this is passed to get_state in order to compute the inputs for the neural network
        # deleted after it has produced a child node based on the current board
        # board can be all the previous and current board states in a list [board0, board1, ...]
        # self.board will be fed into get_state_MCTS in order to get the neural network input

        self.action_history = action_history
        self.current_player = current_player
        self.parent: None or Node or Root = parent

        self.children: list[Node] = [] # a list of node objects which are the children aka resulting future actions
        self.child_legal_actions = child_legal_actions # a list of actions, [action1, action2, ...], will be deleted when completely popped
        self.child_visits = np.zeros(len(child_legal_actions), dtype=np.int32) # not sure if float32 or uint32 would be better for speed
        self.child_values = np.zeros(len(child_legal_actions), dtype=np.float32)


        # Pertaining to the input and outputs of the neural network
        self.RNN_state = RNN_state
        self.child_prob_priors = child_prob_priors

        self.is_terminal = is_terminal # this is the winning player None for not winning node, -1 and 1 for win, 0 for draw

    def __repr__(self):
        return f"{self.action_history, self.is_terminal}"
class Root(Node): # inheritance
    __slots__ = "visits"
    def __init__(self,
                 board: np.array,
                 action_history: list or tuple or np.array,
                 current_player: int,
                 child_legal_actions: list[tuple[int]] or list[int],
                 RNN_state: list or list[np.array] or np.array,
                 child_prob_priors: np.array):
        super().__init__(0,
                         board,
                         action_history,
                         current_player,
                         child_legal_actions,
                         RNN_state,
                         child_prob_priors,
                         is_terminal=None,
                         parent=None)
        # root's child_id will always be 0 because it is not needed
        self.visits = 0
        # don't need value because it isn't needed in PUCT calculations
        del self.child_id # saved 24 bytes OMG

class MCTS:
    # will start from point where the game class left off
    # will update itself after every move assuming the methods are called in the right order
    def __init__(self,
                 game: Gomoku, # the annotation is for testing and debugging
                 c_puct_init: float=2.5,
                 c_puct_base: float=19_652,
                 use_dirichlet=True,
                 dirichlet_alpha=1.11,
                 dirichlet_epsilon=0.25,  # don't change this value, its the weight of exploration noise
                 tau=1.0,
                 fast_find_win=False, # this is for training, when exploiting change to True
                 use_njit=True):
        """
        :param game: Your game class
        :param c_puct_init: Increase this to increase the exploration, too much can causes divergence
        :param c_puct_base: Don't touch, it's the value that determine how much alpha zero explores deeper down the tree
        :param use_dirichlet: To use dirichlet exploration noise or not
        :param dirichlet_alpha: The exploration rate in the alpa zero paper, should be (average moves per game / 10)
        :param dirichlet_epsilon: The weighting that dirichlet noise has over the policy
        :param tau: Temperature which controls how a move is chosen, input tau to be a very small number to chose the move with highest probability
        :param fast_check_win: To uses fast check win parameter in check_win_MCTS
        :param use_njit: jit the method get_terminal_actions
        """
        self.game = game
        self.fast_find_win = fast_find_win

        self.c_puct_init = c_puct_init # determined experimentally
        self.c_puct_base = c_puct_base # DO NOT CHANGE

        self.use_dirichlet = use_dirichlet
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        # dirichlet_alpha can be choosen with (10 / average_moves_per_game), 10 / 9 = 1.11 for tic tac toe
        # see https://ai.stackexchange.com/questions/25939/alpha-zero-does-not-converge-for-connect-6-a-game-with-huge-branching-factor
        # for more info on what c_puct should be along with how dirichlet alpha should be calculated

        self.tau = tau # temperature parameter in alpha zero

        self.use_njit = use_njit
        # Note that these are functions getting assigned and jitted
        if self.use_njit:
            self.get_terminal_actions = nb.njit(self.get_terminal_actions_fn, cache=True, nogil=True)
        else:
            self.get_terminal_actions = self.get_terminal_actions_fn
        # perform inference call to initialize root

        self.create_expand_root()


    @staticmethod
    @njit(cache=True, fastmath=True)
    def _get_best_PUCT_score_index(child_prob_priors: np.array,
                                   child_values: np.array,
                                   child_visits: np.array,
                                   parent_visits: float,
                                   c_puct_init: float,
                                   c_puct_base: float):
        # note that np.log is actually a math ln with base e (2.7)
        U = c_puct_init * child_prob_priors * ((parent_visits ** 0.5) / (child_visits + 1)) * (c_puct_init + np.log((parent_visits + c_puct_base + 1) / c_puct_base))
        PUCT_score = (child_values / child_visits) + U
        return np.argmax(PUCT_score)

    def _PUCT_select(self) -> Node:
        node = self.root
        parent_visits = self.root.visits

        # heavily optimized code
        while True:
            if node.children and node.children[0].is_terminal is not None: # terminal parent
                if np.sum(node.child_values) != 0: # meaning we have winning moves, if it was 0 then all the moves were a draw
                    terminal_nodes = [terminal_child for terminal_child in node.children if terminal_child.is_terminal != 0]
                else:
                    terminal_nodes = node.children
                # THIS RETURNS A CHILD
                return terminal_nodes[np.random.permutation(len(terminal_nodes))[0]]

            if len(node.child_values) > len(node.children):  # we have to select a node
                # and return as it is a new unvisited node
                # THIS RETURNS THE PARENT OF THE CHOSEN CHILD
                return node  # note that this returns the parent, _expand will create the child node


            # expensive call here, use only if PUCT_scores are needed and useful
            best_index = self._get_best_PUCT_score_index(node.child_prob_priors,
                                                         node.child_values,
                                                         node.child_visits,
                                                         parent_visits,
                                                         self.c_puct_init,
                                                         self.c_puct_base)

            # change the node pointer to the selected child at best_index
            parent_visits = node.child_visits[best_index]
            node: Node = node.children[best_index]


    def _compute_outputs(self, input_state, RNN_state):
        # I'll do this once I have everything else working
        # this returns the policy, value, RNN_state in a list
        return self._get_dummy_outputs(input_state, RNN_state)
    def _get_dummy_outputs(self, input_state, RNN_state):
        # since I'm not using RNN_state I can just return it for the next node
        # this will also give the next RNN_state that is required for the next inference call
        if RNN_state is None:
            raise RuntimeError("RNN state cannot be None")
        return np.random.normal(loc=1, scale=1, size=self.game.policy_shape), np.random.uniform(low=-1, high=1, size=(1,))[0], RNN_state # policy and value
        # return np.ones(self.game.policy_shape) / int(self.game.policy_shape[0]), 0.0, RNN_state
    def _apply_dirichlet(self, legal_policy):
        return (1 - self.dirichlet_epsilon) * legal_policy + self.dirichlet_epsilon * np.random.dirichlet(self.dirichlet_alpha * np.ones_like(legal_policy))

    @staticmethod
    # @njit(cache=True)
    def get_terminal_actions_fn(legal_actions_fn,
                             do_action_fn,
                             check_win_fn,
                             board,
                             current_player,
                             fast_find_win=False) -> (np.array, np.array):
        """
        :param board: The board
        :param current_player: Current player we want to check for
        :param WIDTH: board width
        :param HEIGHT: board height
        :param fast_find_win: only returns 1 winning move if True for speed
        This should be False for training, because we want multiple winning moves to determine a better policy
        with more than 1 terminal move
        :return:
        """
        legal_actions = legal_actions_fn(board)
        # reverse the order of each element from [y, x] -> [x, y]

        terminal_actions = []  # includes winning and drawing actions
        terminal_mask = []  # a list of 0 and 1
        # where each index corresponds to a drawing action if 0, and a winning action if 1

        for legal_action in legal_actions:
            # Try every legal action anc check if the current player won
            # Very inefficient. There is a better implementation
            # for simplicity this will be the example

            check_win_board = do_action_fn(board.copy(), legal_action, current_player)

            result = check_win_fn(check_win_board, legal_action, current_player)

            if result != -2:  # this limits the checks by a lot
                terminal_actions.append(
                    legal_action)  # in any case as long as the result != -2, we have a terminal action
                if result == current_player:  # found a winning move
                    terminal_mask.append(1)
                    if fast_find_win:
                        break
                elif result == 0:  # a drawing move
                    terminal_mask.append(0)
        return terminal_actions, np.array(terminal_mask)

    def create_expand_root(self):
        terminal_actions, terminal_mask = self.get_terminal_actions(self.game.get_legal_actions_MCTS,
                                                                    self.game.do_action_MCTS,
                                                                    self.game.check_win_MCTS,
                                                                    self.game.board,
                                                                    self.game.get_current_player(),
                                                                    fast_find_win=self.fast_find_win)

        if terminal_actions:
            if 1 in terminal_mask:
                value = 1
                child_policy = terminal_mask / len(terminal_mask)
            else:
                value = 0
                child_policy = np.ones(len(terminal_mask)) / len(terminal_mask)
            self.root = Root(self.game.board.copy(),
                             self.game.action_history.copy(),
                             self.game.get_current_player() * -1,
                             terminal_actions,
                             [],
                             child_policy)

            del self.root.board
            del self.root.child_legal_actions
            for terminal_action, mask_value in zip(terminal_actions, terminal_mask):
                terminal_player = self.game.get_current_player()
                terminal_node = Node(len(self.root.children),
                self.game.do_action_MCTS(self.game.board.copy(), terminal_action, terminal_player),
                                    self.game.action_history + [terminal_action],
                                    self.game.get_current_player(),
                                    [],
                                    [],
                                    [],
                                    terminal_player if mask_value == 1 else 0,
                                     parent=self.root)
                del terminal_node.child_legal_actions, terminal_node.RNN_state, terminal_node.child_prob_priors
                self.root.children.append(terminal_node)

                self._back_propagate(self.root, value)


        else:
            child_policy, child_value, initial_RNN_state = self._compute_outputs(self.game.board.copy(), [1, 3,3])
            legal_actions, child_prob_prior = self.game.get_legal_actions_policy_MCTS(self.game.board, child_policy)
            if self.use_dirichlet:
                child_prob_prior = self._apply_dirichlet(child_prob_prior)
            self.root = Root(self.game.board.copy(),
                             self.game.action_history.copy(),
                             self.game.get_current_player() * -1, #player for no moves placed
                             list(legal_actions),
                             initial_RNN_state,
                             child_prob_prior)
    def _expand_with_terminal_actions(self, node, terminal_parent_board, terminal_parent_action, terminal_actions, terminal_mask):
        # winning actions must have at least 1 winning action
        len_terminal_moves = len(terminal_actions)
        if 1 in terminal_mask: # if there is a win, then draw's probability should stay at 0
            # ^ this is essentially O(1), because for any NORMAL connect N game there usually is only 1-5 possible ways to win
            terminal_parent_value = 1
            terminal_parent_prob_prior = terminal_mask / len_terminal_moves
        else: # there are only draws
            terminal_parent_value = 0
            terminal_parent_prob_prior = np.ones(len_terminal_moves) / len_terminal_moves # winning policy
            # ^ this is just a formality, it really not needed, but when getting the stats
            # it's nice to see the some numbers that state that there is a win or loss
        terminal_parent_current_player = node.current_player * -1

        terminal_parent = Node(len(node.children),
                               terminal_parent_board,
                               node.action_history + [terminal_parent_action],
                               terminal_parent_current_player,
                               child_legal_actions=terminal_actions,
                               RNN_state=None,
                               child_prob_priors=terminal_parent_prob_prior,
                               is_terminal=None,
                               parent=node)
        node.children.append(terminal_parent)
        del terminal_parent.child_legal_actions

        terminal_parent.child_values = terminal_mask # 0 for draws and 1 for wins, thus perfect for child_values

        terminal_parent.child_visits = np.ones(len(terminal_mask), dtype=np.float32) # formality so that we don't get division by 0 when calcing stats

        for terminal_action, mask_value in zip(terminal_actions, terminal_mask):
            # terminal_board = self.game.do_action_MCTS(child_board, terminal_action)
            # ^ isn't needed because we don't need to use it for the children since is terminal

            terminal_child = Node(child_id=len(terminal_parent.children),
                                  board=None,
                                  action_history=terminal_parent.action_history + [terminal_action],
                                  current_player=node.current_player, # based on node.current_player because node's child's child is the same player as node
                                  child_legal_actions=[1],
                                  RNN_state=None,
                                  child_prob_priors=None,
                                  is_terminal=node.current_player if mask_value == 1 else 0,
                                  parent=terminal_parent)
            # ^ node's child's child

            #deleting stuff because terminal nodes should have no children and thus any stats for the children
            del terminal_child.board
            del terminal_child.children
            del terminal_child.child_values
            del terminal_child.child_visits
            del terminal_child.child_prob_priors
            del terminal_child.child_legal_actions
            del terminal_child.RNN_state

            terminal_parent.children.append(terminal_child)

        return terminal_parent, terminal_parent_value
        # negative because the child's POV won, thus the parent's POV lost in this searched path
        # don't backprop child as there could be multiple ways to win, but all backprop only cares
        # if someone wins


    def _expand(self, node: Node) -> (Node, float):
        # note that node is the parent of the child, and node will always be different and unique
        # create the child to expand
        child_action = node.child_legal_actions.pop(-1) # this must be -1 because list pop is O(1),
        # only when popping from the right

        child_board = self.game.do_action_MCTS(node.board.copy(), child_action, -node.current_player)
        # must copy, because each node child depends on the parent's board state and its action
        # changing the parent's board without copying will cause the parent's board to be changed too
        terminal_actions, terminal_mask = self.get_terminal_actions(self.game.get_legal_actions_MCTS,
                                                                    self.game.do_action_MCTS,
                                                                    self.game.check_win_MCTS,
                                                                    child_board,
                                                                    node.current_player,
                                                                    fast_find_win=self.fast_find_win)

        if terminal_actions:
            child, child_value = self._expand_with_terminal_actions(node, child_board, child_action, terminal_actions, terminal_mask)

        else:
            child_policy, child_value, next_RNN_state = self._compute_outputs(self.game.get_input_state_MCTS(child_board), node.RNN_state)
            # note that child policy is the probabilities for the children of child
            # because we store the policy with the parent rather than in the children
            child_legal_actions, child_prob_prior = self.game.get_legal_actions_policy_MCTS(child_board, child_policy)

            if self.use_dirichlet:
                child_prob_prior = self._apply_dirichlet(child_prob_prior)
            # gets the legal actions and associated probabilities
            child = Node(len(node.children),
                         child_board,
                         node.action_history + [child_action],
                         node.current_player * -1,
                         list(child_legal_actions),
                         next_RNN_state,
                         child_prob_prior,
                         None,
                         parent=node)
            # we don't create every possible child nodes because as the tree gets bigger,
            # there will be more redundant children that do nothing (very unlikely to be visited)
            node.children.append(child)

            if len(node.child_legal_actions) == 0: # we will never use this parent again for generating children
                # because node.children is full
                # and there will no longer be any more legal moves cuz they all have been expanded
                del node.child_legal_actions # saves a lot of memory as we no longer need it

                # since we got the policy and value from inputs node.board
                # we can delete it because the NN will not evaluate it again thus saving more memory
                del node.board
                del node.current_player
                del node.RNN_state # we no longer need RNN state

            # negated child_child_values is the child value

        return child, -child_value # contact Brian if you don't understand why it is -value

    def _back_propagate(self, node: Node, value: float) -> None:
        # note that the node class is the child returned by expand or selection method
        while node.parent is not None:
            # stats are stored in parent
            # so in order to update stats we have to index the parent's
            # child value and child visits
            node_id = node.child_id
            node = node.parent
            # ^ does two things, 1. moves up the thee
            # 2. stores the pointer within in the variable rather than indexing twice
            node.child_values[node_id] += value
            node.child_visits[node_id] += 1
            value *= -1 # negate for the opponent's move
        self.root.visits += 1


    def run(self, iteration_limit: int or bool=None, time_limit: int or float=None):
        """
        :param iteration_limit: The number of iterations MCTS is allowed to run, default will be 5 * number of legal moves
        :param time_limit: The time limit MCTS is allowed to run, note that MCTS can go over the time limit by small amounts
        :return: chosen_action, actions: list, probs: np.array
        where each action in actions corresponds to the prob in probs, actions[0]'s prob is probs[0] and so on
        returns the top action and action distribution for the storage buffer
        """

        if iteration_limit is not True and iteration_limit is not None and iteration_limit < len(self.game.get_legal_actions()):
            warn(f"Iterations must be greater than or equal to {len(self.game.get_legal_actions())}"
                 f"because all depth 1 actions must be visited to produce a valid policy"
                 f"Changing iterations to the default {3 * len(self.game.get_legal_actions())}")

        if (iteration_limit is not None and (iteration_limit is True or iteration_limit < len(self.game.get_legal_actions()))) and time_limit is None:
            iteration_limit = 3 * len(self.game.get_legal_actions())

        if iteration_limit is None and time_limit is True:
            time_limit = 30.0 # 30 seconds by default

        if iteration_limit and time_limit is None:
            bar = tqdm(total=iteration_limit)
        else:
            bar = tqdm(total=time_limit)

        current_iteration = 0
        start_time = time.time()


        while (iteration_limit is None or current_iteration < iteration_limit) and (time_limit is None or time.time() - start_time < time_limit):
            loop_start_time = time.time()
            node = self._PUCT_select()

            if node.is_terminal is not None:
                value = 1 if (node.is_terminal == 1 or node.is_terminal == -1) else 0
            else:
                node, value = self._expand(node)


            self._back_propagate(node, value)

            if iteration_limit:
                bar.update(1)
            else:
                bar.update(time.time() - loop_start_time)
            current_iteration += 1
        bar.close()



        move_probs = [0] * len(self.root.children) # this speeds things up by a bit, compared to append
        for child_id, (child, prob, winrate, value, visits, prob_prior) in enumerate(zip(self.root.children,
                                                                                         self.root.child_visits / self.root.visits,# new probability value
                                                                                         self.root.child_values / self.root.child_visits, # winrate
                                                                                         self.root.child_values,
                                                                                         self.root.child_visits,
                                                                                         self.root.child_prob_priors)):
            move_probs[child_id] = [child.action_history[-1], prob, winrate, value, visits, prob_prior, self.root.visits, child.is_terminal]

        prob_weights = (self.root.child_visits / self.root.visits) ** (1.0 / self.tau)
        prob_weights /= np.sum(prob_weights) # normalize back into a probability distribution
        chosen_index = np.random.choice(np.arange(len(move_probs)), size=1, replace=False, p=prob_weights)[0]
        move = move_probs[chosen_index][0]
        # stochastically sample a move with the weights affected by tau

        move_probs = sorted(move_probs, key=lambda x: x[4], reverse=True)

        return move, move_probs
        # return move, list(map(lambda x: x[:2], move_probs))


    def _set_root(self, child: Node):
        # cannot delete action or child_legal_moves because there is a chance that they haven't been fully searched
        # don't worry they will be deleted in _expand when the time is right

        if child.is_terminal is None:
            if len(child.children) < len(child.child_visits):
                child_legal_actions = child.child_legal_actions
                child_current_player = child.current_player
                child_RNN_state = child.RNN_state
            else:
                child_legal_actions = [0]  # this is only used as an initialization, will be overwritten
                child_current_player = None
                child_RNN_state = None
            child_prob_priors = child.child_prob_priors
        else:
            child_legal_actions = [0] # this is only used as an initialization, will be overwritten
            child_prob_priors = None
            child_current_player = None
            child_RNN_state = None

        new_root = Root(board=self.game.board.copy(),
                        action_history=child.action_history,
                        current_player=child_current_player,
                        child_legal_actions=child_legal_actions,
                        RNN_state=child_RNN_state,
                        child_prob_priors=child_prob_priors)

        # del new_root.RNN_state # don't need this as child must be evaluated
        if child.is_terminal is None:
            new_root.children = child.children
            new_root.child_values = child.child_values
            new_root.child_visits = child.child_visits

            if len(child.children) == len(child.child_visits):
                del new_root.child_legal_actions, new_root.RNN_state, new_root.current_player

        if child.is_terminal is not None:
            del new_root.children, new_root.child_visits, new_root.child_values, new_root.child_prob_priors,  new_root.child_legal_actions, new_root.RNN_state, new_root.current_player

        new_root.visits = self.root.child_visits[child.child_id]
        self.root = new_root

        # terminals = ""
        # for node in self.root.children:
        #     if node.is_terminal is not None:
        #         terminals += f"{node.is_terminal}"
        # print(CRED + terminals + CEND)

        # if self.root.children:
        #     print(self.root.children[0].current_player)

    def prune_tree(self, action):
        # given the move set the root to the child that corresponds to the move played
        # then call set root as root is technically a different class from Node
        for child in self.root.children:
            if np.array_equal(child.action_history[-1], action):
                self._set_root(child)
                return

        # this assumes that the tree was initialized with the other person's perspective
        # and calling prune_tree wants MCTS to play from the opponents perspective,
        # this also assumes that self.game is already updated with the first player's move
        # thus we create a root and expand it
        self.create_expand_root()



if __name__ == "__main__":
    from Guide import Gomoku
    game = Gomoku()
    # game.do_action((7, 7))
    # game.do_action((6, 7))
    # game.do_action((7, 6))
    # game.do_action((6, 6))
    # game.do_action((7, 5))
    #
    # game.do_action((6, 5))

    # game.do_action((7, 4))
    # game.do_action((6, 4))
    mcts = MCTS(game,
                c_puct_init=2.5,
                tau=0.01,
                use_dirichlet=True,
                fast_find_win=True)

    print(game.board)
    winner = -2
    while winner == -2:
        if game.get_current_player() == -1:
            move = game.input_action()
            # print("You played", move)

            game.do_action(move)
            print(game.board)
            mcts.prune_tree(move)
            winner = game.check_win()
        else:
            move, probs = mcts.run(iteration_limit=75000, time_limit=None)
            game.do_action(move)
            print("AI played", move)
            print(probs)
            print(game.board)

            mcts.prune_tree(move)
            winner = game.check_win()
    print("player", winner, "won")

