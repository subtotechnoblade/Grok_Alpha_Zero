import numpy as np
from numba import njit
class Node:
    __slots__ = "board", "action", "parent", "children", "child_legal_actions", "child_visits", "child_values", "child_prob_priors"
    def __init__(self, board : np.array,
                 action : tuple,
                 child_legal_actions:list[tuple],
                 child_prob_priors: np.array,
                 parent=None):
        self.board = board # board, this is passed to get_state in order to compute the inputs for the neural network
        # deleted after it has produced a child node based on the current board
        # board can be all the previous and current board states in a list [board0, board1, ...]
        # self.board will be fed into get_state_MCTS in order to get the neural network input

        self.action = action
        self.parent = parent

        self.children : list[Node] = [] # a list of node objects which are the children aka resulting future actions
        self.child_legal_actions = child_legal_actions # a list of actions, [action1, action2, ...], will be deleted when completely poped
        self.child_visits = np.zeros(len(child_legal_actions), dtype=np.float32)
        self.child_values = np.zeros(len(child_legal_actions), dtype=np.float32)
        self.child_prob_priors = child_prob_priors

class Root(Node): # inheritance
    def __init__(self, board: np.array, action: tuple, action_history: list, child_legal_moves: list[tuple],
                 child_prob_priors: np.array, parent=None):
        super().__init__(board, action, action_history, child_legal_moves, child_prob_priors, parent)
        self.visits = 0
        # don't need value because it isn't needed in PUCT calculations


class MCTS:
    def __init__(self, game,
                 c_puct : float=2.5):
        self.game = game
        self.c_puct = c_puct

    @staticmethod
    @njit(cache=True)
    def _get_best_PUCT_score_index(child_prob_priors: np.array,
                                   child_values: np.array,
                                   child_visits: np.array,
                                   parent_visits: float,
                                   c_puct: float):
        PUCT_score = (child_values / child_visits) + c_puct * child_prob_priors * ((child_visits ** 0.5) / (parent_visits + 1))
        return np.argmax(PUCT_score)

    def _PUCT_select(self, root: Node):
        node = root
        parent_visits = root.visits

        while True:
            # heavily optimized code
            if len(node.child_values) > len(node.children): # we have to select a node to return as it is a new unvisited node
                return node # note that this returns the parent, _expand will create the child node

            best_index = self._get_best_PUCT_score_index(node.child_prob_priors,
                                                         node.child_values,
                                                         node.child_visits,
                                                         parent_visits,
                                                         self.c_puct)

            # change the node pointer to the selected child at best_index
            parent_visits = node.child_visits[best_index]
            node = node.children[best_index]


    def compute_policy_value(self, state):
        # I'll do this once I have everything else working
        pass
    def get_dummy_policy_value(self, state):

        return np.random.normal((3, 3)), np.random.random() # policy and value
    def _expand(self, node: Node) -> Node:
        # note that node is the parent of the child
        # create the child to expand
        child_action = node.child_legal_actions.pop(-1) # this must be -1 because list pop is O(1) only poping from the right
        if len(node.child_legal_actions) == 0:
            del node.child_legal_actions # saves a lot of memory as we no longer need it
        child_policy, child_value = self.get_dummy_policy_value(self.game.get_state_MCTS(node.board))

        child_board = self.game.do_action_MCTS(node.board, child_action)
        child_legal_actions = self.game.get_legal_moves()
        child = Node(child_board, child_action, )


    def _back_prop(self, node):
        pass


