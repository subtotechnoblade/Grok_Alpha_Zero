import numpy as np
from numba import njit
from Guide import Gomoku
class Node:
    __slots__ = "child_id", "board", "action", "current_player", "children", "child_legal_actions", "child_visits", "child_values", "child_prob_priors", "is_terminal",  "parent"
    def __init__(self,
                 child_id: int,
                 board: np.array,
                 action: tuple or int,
                 current_player: int,
                 child_legal_actions:list[tuple[int]] or list[int],
                 child_prob_priors: np.array,
                 is_terminal=None,
                 parent=None):
        self.child_id = child_id
        self.board = board # board, this is passed to get_state in order to compute the inputs for the neural network
        # deleted after it has produced a child node based on the current board
        # board can be all the previous and current board states in a list [board0, board1, ...]
        # self.board will be fed into get_state_MCTS in order to get the neural network input

        self.action = action
        self.current_player = current_player
        self.parent: None or Node or Root = parent

        self.children: list[Node] = [] # a list of node objects which are the children aka resulting future actions
        self.child_legal_actions = child_legal_actions # a list of actions, [action1, action2, ...], will be deleted when completely poped
        self.child_visits = np.zeros(len(child_legal_actions), dtype=np.float32)
        self.child_values = np.zeros(len(child_legal_actions), dtype=np.float32)
        self.child_prob_priors = child_prob_priors

        self.is_terminal = is_terminal # this is the winning player None for not winning node, -1 and 1 for win, 0 for draw

class Root(Node): # inheritance
    __slots__ = "visits"
    def __init__(self, board: np.array, action, current_player, child_legal_actions: list[tuple[int]] or list[int],
                 child_prob_priors: np.array):
        super().__init__(0, board, action, current_player, child_legal_actions, child_prob_priors)
        # root's child_id will always be 0 because it is not needed
        self.visits = 0
        del self.child_id # saved 24 bytes OMG
        del self.parent # saved 16 bytes OMG
        # don't need value because it isn't needed in PUCT calculations

class MCTS:
    # will start from point where the game class left off
    # will update itself after every move assuming the methods are called in the right order
    def __init__(self,
                 game: Gomoku, # this is for testing and debugging
                 c_puct_init: float=2.5,
                 c_puct_base: float=19_652,
                 use_dirichlet=True,
                 dirichlet_alpha=1.11,
                 fast_check_win=False # this is for training, when exploiting change to True
                 ):
        self.game = game
        self.fast_check_win = fast_check_win

        self.c_puct_init = c_puct_init # determined experimentally
        self.c_puct_base = c_puct_base # DO NOT CHANGE

        self.use_dirichlet = use_dirichlet
        self.dirichlet_alpha = dirichlet_alpha
        # dirichlet_alpha can be choosen with (10 / average_moves_per_game), 10 / 9 = 1.11 for tic tac toe
        # see https://ai.stackexchange.com/questions/25939/alpha-zero-does-not-converge-for-connect-6-a-game-with-huge-branching-factor
        # for more info on what c_puct should be along with how dirichlet alpha should be calculated

        # perform inference call to initialize root
        child_policy, child_value = self.get_dummy_policy_value(self.game.board.copy())
        legal_actions, child_prob_prior = self.game.get_legal_actions_policy_MCTS(self.game.board, child_policy)
        self.root = Root(game.board.copy(), self.game.action_history[-1], legal_actions, child_prob_prior)

    @staticmethod
    # @njit(cache=True)
    def _get_best_PUCT_score_index(child_prob_priors: np.array,
                                   child_values: np.array,
                                   child_visits: np.array,
                                   parent_visits: float,
                                   c_puct_init: float,
                                   c_puct_base: float):
        # note that np.log is actually a math ln with base e (2.7)
        U = child_prob_priors * ((child_visits ** 0.5) / (parent_visits + 1)) * (c_puct_init + np.log((parent_visits + c_puct_base + 1) / c_puct_base))
        PUCT_score = (child_values / child_visits) + U
        return np.argmax(PUCT_score)

    def _PUCT_select(self, root: Root) -> Node:
        node = root
        parent_visits = root.visits

        # heavily optimized code
        while True:
            if node.children and node.children[0].is_terminal is not None: # terminal parent
                if np.sum(node.child_values) != 0: # meaning we have winning moves, if it was 0 then all the moves were a draw
                    terminal_nodes = [terminal_child for terminal_child in node.children if terminal_child.is_terminal != 0]
                else:
                    terminal_nodes = node.children
                return terminal_nodes[np.random.randint(1, len(terminal_nodes) + 1)] # + 1 because upper is not inclusive
            
            
            if len(node.child_values) > len(node.children): # we have to select a node to return as it is a new unvisited node
                return node # note that this returns the parent, _expand will create the child node

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


    def compute_policy_value(self, state):
        # I'll do this once I have everything else working
        pass
    def get_dummy_policy_value(self, state):

        return np.random.normal((9,)), np.random.random() # policy and value

    def apply_dirichlet(self, legal_policy):
        # I'll implement this layer
        pass

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

        terminal_parent = Node(len(node.children), terminal_parent_board, terminal_parent_action, terminal_parent_current_player, [], terminal_parent_prob_prior,
                     None, parent=node)
        node.append(terminal_parent)

        terminal_parent.child_values = terminal_mask # 0 for draws and 1 for wins, thus perfect for child_values

        terminal_parent.child_visits = np.ones(len(terminal_mask), dtype=np.float32) # formality so that we don't get division by 0 when calcing stats

        for terminal_action, mask_value in zip(terminal_actions, terminal_mask):
            # terminal_board = self.game.do_action_MCTS(child_board, terminal_action)
            # ^ isn't needed because we don't need to use it for the children since is terminal

            # based on node.current_player because node's child's child is the same player as node
            terminality = node.current_player if mask_value == 1 else 0
            terminal_child = Node(len(terminal_parent.children), None, terminal_action, node.current_player, [], [], terminality, parent=terminal_parent)
            # ^ node's child's child

            #deleting stuff because terminal nodes should have no children and thus any stats for the children
            del terminal_child.board
            del terminal_child.children
            del terminal_child.child_values
            del terminal_child.child_visits
            del terminal_child.child_prob_priors
            del terminal_child.child_legal_actions

            terminal_parent.children.append(terminal_child)

        return terminal_parent, terminal_parent_value


    def _expand(self, node: Node) -> tuple[Node, float]:
        # note that node is the parent of the child, and node will always be different and unique
        # create the child to expand
        child_action = node.child_legal_actions.pop(-1) # this must be -1 because list pop is O(1),
        # only when popping from the right
        child_board = self.game.do_action_MCTS(node.board, child_action)

        terminal_actions, terminal_mask = self.game.get_terminal_actions_MCTS(child_board, node.current_player * -1, fast_check=self.fast_check_win)
        if terminal_actions:
            child, child_value = self._expand_with_terminal_actions(node, child_board, child_action, node.current_player, terminal_actions, terminal_mask)

        else:
            child_policy, child_value = self.get_dummy_policy_value(self.game.get_state_MCTS(child_board))
            # note that child policy is the probabilities for the children of child
            # because we store the policy with the parent rather than in the children
            child_legal_actions, child_prob_prior = self.game.get_legal_actions_policy_MCTS(child_board, child_policy)
            # gets the legal actions and associated probabilities
            child = Node(len(node.children), child_board, child_action, node.current_player * -1, child_legal_actions, child_prob_prior, None, parent=node)
            # we dont't create every possible child nodes because as the tree gets bigger,
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
            # negated child_child_values is the child value
        return child, -child_value # contact Brian if you don't understand why it is -value

    def _back_propagate(self, node):
        pass

    def run(self):
        # will probably need to update requirements.txt for tqdm as a new library
        # returns the top action and action distribution for the storage buffer
        pass

    def _set_root(self, child: Node):
        # cannot delete action or child_legal_moves because there is a chance that they haven't been fully searched
        # don't worry they will be deleted in _expand when the time is right
        new_root = Root(child.board, child.action, child.child_legal_actions, child.child_prob_priors)
        new_root.children = child.children
        new_root.visits = self.root.child_visits[child.child_id]
        self.root = new_root
    def prune_tree(self, action):
        # given the move set the root to the child that corresponds to the move played
        # then call set root as root is technically a different class from Node
        for child in self.root.children:
            if child.action == action:
                self._set_root(child)
                break



