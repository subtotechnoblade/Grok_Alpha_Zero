import numpy as np
from numba import njit
class Node:
    __slots__ = "child_id", "board", "action", "parent", "children", "child_legal_actions", "child_visits", "child_values", "child_prob_priors", "is_terminal"
    def __init__(self,
                 child_id: int,
                 board : np.array,
                 action : tuple or int,
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
        self.parent: None or Node or Root = parent

        self.children: list[Node] = [] # a list of node objects which are the children aka resulting future actions
        self.child_legal_actions = child_legal_actions # a list of actions, [action1, action2, ...], will be deleted when completely poped
        self.child_visits = np.zeros(len(child_legal_actions), dtype=np.float32)
        self.child_values = np.zeros(len(child_legal_actions), dtype=np.float32)
        self.child_prob_priors = child_prob_priors

        self.is_terminal = is_terminal # this is the winning player None for not winning node, -1 and 1 for win, 0 for draw

class Root(Node): # inheritance
    __slots__ = "visits"
    def __init__(self, board: np.array, action, child_legal_actions: list[tuple[int]] or list[int],
                 child_prob_priors: np.array):
        super().__init__(0, board, action, child_legal_actions, child_prob_priors)
        # root's child_id will always be 0 because it is not needed
        self.visits = 0
        del self.child_id # saved 24 bytes OMG
        del self.parent # saved 16 bytes OMG
        # don't need value because it isn't needed in PUCT calculations


class MCTS:
    # will start from point where the game class left off
    # will update itself after every move assuming the methods are called in the right order
    def __init__(self,
                 game,
                 c_puct: float=2.5,
                 use_dirichlet=True,
                 dirichlet_alpha=0.3,):
        self.game = game
        self.c_puct = c_puct
        self.use_dirichlet = use_dirichlet
        self.dirichlet_alpha = dirichlet_alpha

        # perform inference call to initialize root
        child_policy, child_value = self.get_dummy_policy_value(self.game.board.copy())
        legal_actions, child_prob_prior = self.game.get_legal_actions_policy_MCTS(self.game.board, child_policy)
        self.root = Root(game.board.copy(), self.game.action_history[-1], legal_actions, child_prob_prior)

    @staticmethod
    @njit(cache=True)
    def _get_best_PUCT_score_index(child_prob_priors: np.array,
                                   child_values: np.array,
                                   child_visits: np.array,
                                   parent_visits: float,
                                   c_puct: float):
        PUCT_score = (child_values / child_visits) + c_puct * child_prob_priors * ((child_visits ** 0.5) / (parent_visits + 1))
        return np.argmax(PUCT_score)

    def _PUCT_select(self, root: Root) -> Node:
        node = root
        parent_visits = root.visits

        while True:
            # heavily optimized code
            if len(node.child_values) > len(node.children) or node.is_terminal is None: # we have to select a node to return as it is a new unvisited node
                return node # note that this returns the parent, _expand will create the child node

            best_index = self._get_best_PUCT_score_index(node.child_prob_priors,
                                                         node.child_values,
                                                         node.child_visits,
                                                         parent_visits,
                                                         self.c_puct)

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

    def _expand(self, node: Node) -> tuple[Node, float]:
        # note that node is the parent of the child, and node will always be different and unique
        # create the child to expand
        child_action = node.child_legal_actions.pop(-1) # this must be -1 because list pop is O(1) only poping from the right

        child_policy, child_value = self.get_dummy_policy_value(self.game.get_state_MCTS(node.board))

        child_board = self.game.do_action_MCTS(node.board, child_action)
        child_legal_actions, child_prob_prior = self.game.get_legal_actions_policy_MCTS(child_board, child_policy)
        # gets the legal actions and associated probabilities
        child = Node(len(node.children), child_board, child_action, child_legal_actions, child_prob_prior, parent=node)
        node.children.append(child)

        if len(node.child_legal_actions) == 0: # we will never use this parent again because children is full
            # and there will no longer be any more legal moves cuz they all have been expanded
            del node.child_legal_actions # saves a lot of memory as we no longer need it

            # since we got the policy and value from inputs node.board
            # we can delete it because the NN will not evaluate it again thus saving more memory
            del node.board
            del node.action # we don't need this anymore because check win doesn't require this
        return child, child_value # for value backpropagation

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



