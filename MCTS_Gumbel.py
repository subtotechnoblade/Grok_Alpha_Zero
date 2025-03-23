import os
import time

import numpy as np

np.seterr(all='raise')
import numba as nb
from numba import njit
from numba.extending import is_jitted

from Session_Cache import Cache_Wrapper
from tqdm import tqdm
from warnings import warn

import onnxruntime as rt
from Client_Server import Parallelized_Session


class Node:
    __slots__ = "child_id", "board", "action_history", "current_player", "children", "child_legal_actions", "child_visits", "child_values", "RNN_state", "child_logit_priors", "is_terminal", "parent"

    def __init__(self,
                 child_id: int,
                 board: np.array,
                 action_history: np.array,
                 current_player: int,
                 child_legal_actions: list[tuple[int]] or list[int],
                 RNN_state: list or list[np.array, ...] or np.array,
                 child_logit_priors: np.array,
                 is_terminal=None,
                 parent=None):
        self.child_id = child_id
        self.board = board  # board, this is passed to get_state in order to compute the inputs for the neural network
        # deleted after it has produced a child node based on the current board
        # board can be all the previous and current board states in a list [board0, board1, ...]
        # self.board will be fed into get_state_MCTS in order to get the neural network input

        self.action_history = action_history
        self.current_player = current_player
        self.parent: None or Node or Root = parent

        self.children: list[Node] or list[None] = [None for _ in range(
            len(child_legal_actions))]  # a list of node objects which are the children aka resulting future actions
        self.child_legal_actions = child_legal_actions  # a list of actions, [action1, action2, ...], will be deleted when completely popped
        self.child_visits = np.zeros(len(child_legal_actions),
                                     dtype=np.uint32)  # not sure if float32 or uint32 would be better for speed
        self.child_values = np.zeros(len(child_legal_actions), dtype=np.float32)

        # Pertaining to the input and outputs of the neural network
        self.RNN_state = RNN_state
        self.child_logit_priors = child_logit_priors

        self.is_terminal = is_terminal  # this is the winning player None for not winning node, -1 and 1 for win, 0 for draw


class Root(Node):  # inheritance
    __slots__ = "visits"

    def __init__(self,
                 board: np.array,
                 action_history: list or tuple or np.array,
                 current_player: int,
                 child_legal_actions: list[tuple[int]] or list[int] or dict[int:np.ndarray],
                 RNN_state: list or list[np.array] or np.array,
                 child_logit_priors: np.array):
        super().__init__(0,
                         board,
                         action_history,
                         current_player,
                         child_legal_actions,
                         RNN_state,
                         child_logit_priors,
                         is_terminal=None,
                         parent=None)
        # root's child_id will always be 0 because it is not needed
        self.visits = 0
        # don't need value because it isn't needed in PUCT calculations
        del self.child_id  # saved 24 bytes OMG


@njit("float32[:](float32[:])", cache=True)
def stablemax(logits):
    s_x = np.where(logits >= 0, logits + 1, 1 / (1 - logits))
    return s_x / np.sum(s_x)


@njit("float32[:](float32[:])", cache=True)
def softmax(logits):
    exp = np.exp(logits)
    return exp / np.sum(logits)


@njit(["float32[:](float32[:], uint32[:], float32, float32)",
       "float32[:](float32[:], int64, float32, float32)"]
    , cache=True, fastmath=True)
def q_transform(input_values, visits, min_value=-1.0, max_value=1.0):
    values = np.where(visits > 0, input_values, min_value)
    return (values - min_value) / (max_value - min_value)


@njit("float32[:](float32[:], float32, float32, float32)", cache=True, fastmath=True)
def sigma(q, N_b, c_visit, c_scale):
    return (c_visit + N_b) * c_scale * q


@njit(cache=True)
def compute_pi(values,
               logits,
               visits,
               N_b,  # max visits for any action (this will be a child of the root)
               c_visit,
               c_scale,
               use_softmax=True,
               min_value=-1.0,
               max_value=1.0,
               ):
    q = q_transform(values, visits, min_value, max_value)
    if use_softmax:
        completed_q = np.where(visits > 0, q, np.sum(softmax(logits) * q))
        return softmax(logits + sigma(completed_q, N_b, c_visit, c_scale))
    else:
        completed_q = np.where(visits > 0, q, np.sum(stablemax(logits) * q))
        return stablemax(logits + sigma(completed_q, N_b, c_visit, c_scale))


class MCTS_Gumbel:
    # will start from point where the game class left off
    # will update itself after every move assuming the methods are called in the right order
    def __init__(self,
                 game,  # the annotation is for testing and debugging
                 session: rt.InferenceSession or Parallelized_Session or Cache_Wrapper or None,
                 use_njit=None,
                 m=16,
                 c_visit=50.0,
                 c_scale=1.0,
                 activation_fn="stablemax",
                 fast_find_win=False,  # this is for training, when exploiting change to True
                 ):
        """
        :param game: Your game class
        :param c_puct_init: Increase this to increase the exploration, too much can causes divergence
        :param c_puct_base: Don't touch, it's the value that determine how much alpha zero explores deeper down the tree
        :param use_dirichlet: To use dirichlet exploration noise or not
        :param dirichlet_alpha: The exploration rate in the alpa zero paper, should be (average moves per game / 10)
        :param dirichlet_epsilon: The weighting that dirichlet noise has over the policy
        :param tau: Temperature which controls how a move is chosen, input tau to be a very small number to chose the move with highest probability
        """
        self.game = game
        self.session = session
        self.cache_session = isinstance(self.session, Cache_Wrapper)
        self.fast_find_win = fast_find_win

        self.use_njit = use_njit if use_njit is not None else hasattr(os, "fork")
        self.m = m
        self.c_visit = c_visit
        self.c_scale = c_scale

        self.use_softmax = activation_fn == "softmax"

        njitable = is_jitted(self.game.get_legal_actions_MCTS) and is_jitted(
            self.game.do_action_MCTS) and is_jitted(self.game.check_win_MCTS)

        # Note that these are functions getting assigned and jitted
        if njitable and self.use_njit:
            self.get_terminal_actions = nb.njit(self.get_terminal_actions_fn, cache=hasattr(os, "fork"))
        else:
            self.get_terminal_actions = self.get_terminal_actions_fn

        # perform inference call to initialize root
        self.create_expand_root()

    def update_hyperparams(self, *args, **kwargs) -> None:
        m = kwargs.get("m")
        if m is not None:
            self.m = m

        c_scale = kwargs.get("c_scale")
        if c_scale is not None:
            self.c_scale = c_scale

        c_visit = kwargs.get("c_visit")
        if c_visit is not None:
            self.c_visit = c_visit

    @staticmethod
    @njit("Tuple((int64[:], int64))(int64, int64, float32[:], float32[:], int64, int64, float32, int64)", cache=True)
    def sequential_halving(m, n, gumbel_logits, q_hat, N_b, c_visit, c_scale, phase, ):
        """
        return the index of the top actions as a generator
        """
        if phase == 0:
            child_ids = np.argsort(gumbel_logits)[-m:]
        else:
            child_ids = np.argsort(gumbel_logits + sigma(q_hat, N_b, c_visit, c_scale))[-int(m / (2 ** phase)):]
        visit_budget_per_child = int(n / (np.log2(m) * (m / (2 ** phase))))
        return child_ids, visit_budget_per_child

    @staticmethod
    @njit(cache=True)
    def deterministic_selection(values,
                                logits,
                                visits,
                                N_b,
                                c_visit,
                                c_scale,
                                use_softmax=False,
                                min_value=-1.0,
                                max_value=1.0):
        pi = compute_pi(values, logits, visits, N_b, c_visit, c_scale, use_softmax=use_softmax, min_value=min_value,
                        max_value=max_value)
        return np.argmax(pi - (visits / (1 + np.sum(visits))))

    def select(self, node: Node):
        while True:
            child_id = self.deterministic_selection(node.child_values,
                                                    node.child_logit_priors,
                                                    node.child_visits,
                                                    self.root.child_visits.max(),
                                                    self.c_visit,
                                                    self.c_scale,
                                                    self.use_softmax)
            if node.children[child_id] is None:
                return node, child_id  # Note that the node is the parent not the child
            elif node.children[child_id].is_terminal is not None:
                return node.children[child_id], child_id
            node = node.children[child_id]

    def _compute_outputs(self, inputs, RNN_state, depth=0):
        if self.session is not None:
            # input_state, input_state_matrix = RNN_state
            kwargs = {
                "output_names": ["policy", "value"],
                "input_feed": {"inputs": np.expand_dims(inputs.astype(dtype=np.float32, copy=False), 0)}
            }
            if self.cache_session:
                kwargs["depth"] = depth

            policy, value = self.session.run(**kwargs)
            return policy[0], value[0][0], RNN_state
        return self._get_dummy_outputs(inputs, RNN_state)

    def _get_dummy_outputs(self, input_state, RNN_state):
        # since I'm not using RNN_state I can just return it for the next node
        # this will also give the next RNN_state that is required for the next inference call
        if RNN_state is None:
            raise RuntimeError("RNN state cannot be None")
        # return np.ones(self.game.policy_shape) / self.game.policy_shape[0], 0.0, RNN_state
        return np.random.uniform(low=0, high=1, size=self.game.policy_shape).astype(np.float32, copy=False), \
            np.random.uniform(low=-1, high=1, size=(1,))[0], RNN_state  # policy and value
        # return np.ones(self.game.policy_shape) / int(self.game.policy_shape[0]), 0.0, RNN_state

    @staticmethod
    def get_terminal_actions_fn(do_action_fn,
                                check_win_fn,
                                action_histories,
                                board,
                                next_player,
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
        terminal_index = []  # includes winning and drawing actions
        terminal_mask = []  # a list of 0 and 1
        # where each index corresponds to a drawing action if 0, and a winning action if 1
        for action_id in range(len(action_histories)):
            # Try every legal action and check if the current player won
            # Very inefficient. There is a better implementation

            legal_action = action_histories[action_id][-1]
            check_win_board = do_action_fn(board.copy(), legal_action, next_player)

            result = check_win_fn(check_win_board, next_player, action_histories[action_id])
            if result != -2:  # this limits the checks by a lot
                terminal_index.append(action_id)  # in any case as long as the result != -2, we have a terminal action
                if result == next_player:  # found a winning move
                    terminal_mask.append(1)
                    if fast_find_win:
                        break
                elif result == 0:  # a drawing move
                    terminal_mask.append(0)
        return action_histories[:, -1][np.array(terminal_index, dtype=np.int32)], np.array(terminal_mask,
                                                                                           dtype=np.float32)

    def create_expand_root(self):
        action_history = np.array(self.game.action_history)
        legal_actions = self.game.get_legal_actions_MCTS(self.game.board,
                                                         -self.game.get_next_player(),
                                                         action_history)
        if self.game.action_history:
            repeated_action_history = np.zeros((len(legal_actions), *action_history.shape,), dtype=action_history.dtype)
            repeated_action_history[:] = action_history
            future_action_histories = np.concatenate((repeated_action_history,
                                                      np.expand_dims(legal_actions, 1)), axis=1)
        else:
            future_action_histories = np.expand_dims(legal_actions, 1)
        terminal_actions, terminal_mask = self.get_terminal_actions(self.game.do_action_MCTS,
                                                                    self.game.check_win_MCTS,
                                                                    future_action_histories,
                                                                    self.game.board,
                                                                    self.game.get_next_player(),
                                                                    fast_find_win=self.fast_find_win)

        if len(terminal_actions) > 0:
            if 1 in terminal_mask:
                value = 1
                child_policy = terminal_mask / len(terminal_mask)
            else:
                value = 0
                child_policy = np.ones(len(terminal_mask), dtype=np.float32) / len(terminal_mask)

            self.root = Root(self.game.board.copy(),
                             self.game.action_history.copy(),
                             -self.game.get_next_player(),
                             terminal_actions,
                             [],
                             child_policy)
            self.root.child_legal_actions = {}
            del self.root.board
            for child_id, (terminal_action, mask_value) in enumerate(zip(terminal_actions, terminal_mask)):
                terminal_player = self.game.get_next_player()
                terminal_node = Node(child_id,
                                     self.game.do_action_MCTS(self.game.board.copy(), terminal_action, terminal_player),
                                     self.game.action_history + [terminal_action],
                                     -self.game.get_next_player(),
                                     [],
                                     [],
                                     [],
                                     terminal_player if mask_value == 1 else 0,
                                     parent=self.root)
                del terminal_node.child_legal_actions
                del terminal_node.RNN_state, terminal_node.child_logit_priors
                self.root.children[child_id] = terminal_node
                # print(True)
                self._back_propagate(terminal_node, value)


        else:
            child_policy, child_value, initial_RNN_state = self._compute_outputs(self.game.get_input_state(),
                                                                                 [],
                                                                                 len(self.game.action_history))

            legal_actions, child_logit_prior = self.game.get_legal_actions_policy_MCTS(self.game.board,
                                                                                       -self.game.get_next_player(),
                                                                                       np.array(
                                                                                           self.game.action_history),
                                                                                       child_policy,
                                                                                       normalize=False)

            self.root = Root(self.game.board.copy(),
                             self.game.action_history.copy(),
                             self.game.get_next_player() * -1,  # current player for no moves placed
                             dict(enumerate(legal_actions)),
                             initial_RNN_state,
                             child_logit_prior)

    def _expand_with_terminal_actions(self, node, child_index, terminal_parent_board, terminal_parent_action,
                                      terminal_actions,
                                      terminal_mask):
        # winning actions must have at least 1 winning action

        if 1 in terminal_mask:  # if there is a win, then draw's probability should stay at 0
            # ^ this is essentially O(1), because for any NORMAL connect N game there usually is only 1-5 possible ways to win
            num_winning_actions = len(terminal_mask == 1)  # get the length of thr array that is equal to 1
            terminal_parent_value = num_winning_actions  # the evaluation when we win num_winning_actions
            # terminal_parent_value = 1 # debug
            terminal_parent_visits = num_winning_actions  # we must visits num_winning_actions to win that many times
            # terminal_parent_visits = 1 # debug
            terminal_parent_prob_prior = terminal_mask.astype(np.float32, copy=False) / num_winning_actions
        else:  # there are only draws
            len_terminal_moves = len(terminal_actions)
            terminal_parent_value = 0
            # terminal_parent_visits = 1 # debug
            terminal_parent_visits = len_terminal_moves
            terminal_parent_prob_prior = np.ones(len_terminal_moves,
                                                 dtype=np.float32) / len_terminal_moves  # winning policy
            # ^ this is just a formality, it really not needed, but when getting the stats
            # it's nice to see the some numbers that state that there is a win or loss
        terminal_parent_current_player = node.current_player * -1

        terminal_parent = Node(child_index,
                               terminal_parent_board,
                               node.action_history + [terminal_parent_action],
                               terminal_parent_current_player,
                               child_legal_actions=dict(enumerate(terminal_actions)),
                               RNN_state=None,
                               child_logit_priors=terminal_parent_prob_prior,
                               is_terminal=None,
                               parent=node)
        node.children[child_index] = terminal_parent

        terminal_parent.child_values = terminal_mask.astype(np.float32,
                                                            copy=False)  # 0 for draws and 1 for wins, thus perfect for child_values

        terminal_parent.child_visits = np.ones(len(terminal_mask), dtype=np.uint32)
        # formality so that we don't get division by 0 when calcing stats

        for child_id, (terminal_action, mask_value) in enumerate(zip(terminal_actions, terminal_mask)):
            # terminal_board = self.game.do_action_MCTS(child_board, terminal_action)
            # ^ isn't needed because we don't need to use it for the children since is terminal

            terminal_child = Node(child_id=child_id,
                                  board=None,
                                  action_history=terminal_parent.action_history + [terminal_action],
                                  current_player=node.current_player,
                                  # based on node.current_player because node's child's child is the same player as node
                                  child_legal_actions=[None],
                                  RNN_state=None,
                                  child_logit_priors=None,
                                  is_terminal=node.current_player if mask_value == 1 else 0,
                                  parent=terminal_parent)
            # ^ node's child's child

            # deleting stuff because terminal nodes should have no children and thus any stats for the children
            del terminal_child.board
            del terminal_child.children
            del terminal_child.child_values
            del terminal_child.child_visits
            del terminal_child.child_logit_priors
            del terminal_child.RNN_state
            terminal_parent.children[child_id] = terminal_child

        return terminal_parent, -terminal_parent_value, terminal_parent_visits

        # negative because the child's POV won, thus the parent's POV lost in this searched path
        # don't backprop child as there could be multiple ways to win, but all backprop only cares
        # if someone wins

    def _expand(self, node: Node, index: int or np.uint32) -> (Node, float):
        # note that node is the parent of the child, and node will always be different and unique
        # create the child to expand
        child_action = node.child_legal_actions.pop(index)  # child_legal_actions will now be a dict

        # Note that from now on, -current_player is the current_player for child_board
        child_board = self.game.do_action_MCTS(node.board.copy(), child_action, -node.current_player)
        # must copy, because each node child depends on the parent's board state and its action
        # changing the parent's board without copying will cause the parent's board to be changed too
        action_history = np.array(self.game.action_history)
        legal_actions = self.game.get_legal_actions_MCTS(child_board,
                                                         -node.current_player,
                                                         action_history)
        if self.game.action_history:
            repeated_action_history = np.zeros((len(legal_actions), *action_history.shape,), dtype=action_history.dtype)
            repeated_action_history[:] = action_history
            future_action_histories = np.concatenate((repeated_action_history,
                                                      np.expand_dims(legal_actions, 1)), axis=1)
        else:
            future_action_histories = np.expand_dims(legal_actions, 1)

        terminal_actions, terminal_mask = self.get_terminal_actions(self.game.do_action_MCTS,
                                                                    self.game.check_win_MCTS,
                                                                    future_action_histories,
                                                                    child_board,
                                                                    node.current_player,
                                                                    fast_find_win=self.fast_find_win)

        if len(terminal_actions) > 0:
            return self._expand_with_terminal_actions(node, index, child_board, child_action, terminal_actions,
                                                      terminal_mask)
        else:
            child_policy, child_value, next_RNN_state = self._compute_outputs(
                self.game.get_input_state_MCTS(child_board,
                                               -node.current_player,
                                               np.array(node.action_history + [child_action])),
                node.RNN_state, len(node.action_history))

            # note that child policy is the probabilities for the children of child
            # because we store the policy with the parent rather than in the children
            child_legal_actions, child_logit_prior = self.game.get_legal_actions_policy_MCTS(child_board,
                                                                                             -node.current_player,
                                                                                             np.array(
                                                                                                 node.action_history),
                                                                                             child_policy,
                                                                                             normalize=False)

            child = Node(index,
                         child_board,
                         node.action_history + [child_action],
                         node.current_player * -1,
                         dict(enumerate(child_legal_actions)),
                         next_RNN_state,
                         child_logit_prior,
                         None,
                         parent=node)
            # we don't create every possible child nodes because as the tree gets bigger,
            # there will be more redundant children that do nothing (very unlikely to be visited)
            node.children[index] = child

            if len(node.child_legal_actions) == 0:  # we will never use this parent again for generating children
                # because node.children is full
                # and there will no longer be any more legal moves cuz they all have been expanded

                # since we got the policy and value from inputs node.board
                # we can delete it because the NN will not evaluate it again thus saving more memory
                del node.board
                del node.current_player
                del node.RNN_state  # we no longer need RNN state
            # negated child_child_values is the child value
        return child, -child_value, 1  # contact Brian if you don't understand why it is -value

    def _back_propagate(self, node: Node, value: float, visits=1) -> None:
        # note that the node class is the child returned by expand or selection method
        while node.parent is not None:
            # stats are stored in parent
            # so in order to update stats we have to index the parent's
            # child value and child visits
            node_id = node.child_id
            node = node.parent
            # ^ does two things, 1. moves up the tree
            # 2. stores the pointer within in the variable rather than indexing twice
            node.child_values[node_id] += value
            node.child_visits[node_id] += visits
            value *= -1  # negate for the opponent's move
        self.root.visits += visits

    def fill_empty_children(self):
        for node_id, node in enumerate(self.root.children):
            if node is None:
                child_action = self.root.child_legal_actions.pop(node_id)
                child = Node(node_id,
                             None,
                             self.root.action_history + [child_action],
                             self.root.current_player * -1,
                             [None],
                             None,
                             None,
                             None,
                             self.root)
                self.root.children[node_id] = child

    def run(self, iteration_limit: int or bool = None, time_limit: int or float = None, use_bar=True):
        """
        :param iteration_limit: The number of iterations MCTS is allowed to run, default will be 5 * number of legal moves
        :param time_limit: The time limit MCTS is allowed to run, note that MCTS can go over the time limit by small amounts
        :return: chosen_action, actions: list, probs: np.array
        where each action in actions corresponds to the prob in probs, actions[0]'s prob is probs[0] and so on
        returns the top action and action distribution for the storage buffer
        """
        legal_actions = self.game.get_legal_actions()
        len_legal_actions = len(legal_actions)

        if (iteration_limit is not None and iteration_limit is True) and time_limit is None:
            iteration_limit = len(self.game.get_legal_actions()) * 3

        if time_limit is not None:
            iteration_limit = len(self.game.get_legal_actions()) * 3
            warn("Time limit isn't allowed for gumbel MCTS defaulting to use 3 * len_legal_actions")

        num_starting_legal_actions = len(self.game.get_legal_actions())
        if self.m > num_starting_legal_actions:
            self.m = num_starting_legal_actions
        if use_bar:
            bar = tqdm(total=iteration_limit)

        assert iteration_limit > 0

        current_iteration = 0
        current_phase = 0
        gumbel_noise = np.random.gumbel(loc=0.0, scale=1.0, size=(len(self.root.child_logit_priors),))

        top_gumbel_logits = (self.root.child_logit_priors + gumbel_noise).astype(np.float32, copy=False)
        # the * 20 is for debugging for probs to "simulate" logits
        top_node_ids = np.arange(len(self.root.children))
        top_mean_values = self.root.child_values

        while len_legal_actions > 1:

            chosen_ids, visits_per_child = self.sequential_halving(self.m,
                                                                   iteration_limit,
                                                                   top_gumbel_logits,
                                                                   q_transform(top_mean_values, 1, -1.0, 1.0),
                                                                   self.root.child_visits.max(),
                                                                   self.c_visit,
                                                                   self.c_scale,
                                                                   current_phase)

            chosen_ids = np.sort(chosen_ids)

            top_gumbel_logits = top_gumbel_logits[chosen_ids]
            top_node_ids = top_node_ids[chosen_ids]
            top_values = self.root.child_values[top_node_ids]
            len_root_ids = len(chosen_ids)
            if len_root_ids == 1:
                break

            if len_root_ids == 2 or len_root_ids == 3:
                visits_per_child = (iteration_limit - current_iteration) // len_root_ids

            for root_child_id in top_node_ids:
                if self.root.children[root_child_id] is None:
                    node, value, visits = self._expand(self.root, root_child_id)
                    self._back_propagate(node, value, visits)

                for _ in range(visits_per_child):
                    node: Node = self.root.children[root_child_id]
                    if node.is_terminal is None:
                        node, child_id = self.select(node)

                    if node.is_terminal is not None:
                        value = 1 if (node.is_terminal == 1 or node.is_terminal == -1) else 0
                        visits = 1
                    else:
                        node, value, visits = self._expand(node, child_id)

                    self._back_propagate(node, value, visits)
                    if use_bar:
                        bar.update(1)
                    current_iteration += 1
            top_mean_values = (top_values / self.root.child_visits[top_node_ids]).astype(np.float32, copy=False)
            current_phase += 1

        if use_bar:
            bar.close()

        move_probs = [0] * len(self.root.children)  # this speeds things up by a bit, compared to append
        pi = compute_pi(self.root.child_values,
                        self.root.child_logit_priors,
                        self.root.child_visits,
                        self.root.child_visits.max(),
                        self.c_visit, self.c_scale,
                        self.use_softmax)

        self.fill_empty_children()
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_values = self.root.child_values / self.root.child_visits
            mean_values = np.where(self.root.child_visits > 0, mean_values, pi)
        for child_id, (child, prob, winrate, value, visits, prob_prior) in enumerate(zip(self.root.children,
                                                                                         pi,
                                                                                         mean_values,  # winrate
                                                                                         self.root.child_values,
                                                                                         self.root.child_visits,
                                                                                         self.root.child_logit_priors)):
            move_probs[child_id] = [child.action_history[-1], prob, winrate, value, visits, prob_prior,
                                    self.root.visits, child.is_terminal]
        # stochastically sample a move with the weights affected by tau

        move_probs = sorted(move_probs, key=lambda x: x[1], reverse=True)
        return self.root.children[top_node_ids[0]].action_history[-1], move_probs

    def _set_root(self, child: Node):
        # cannot delete action or child_legal_moves because there is a chance that they haven't been fully searched
        # don't worry they will be deleted in _expand when the time is right

        if child.is_terminal is None:
            if len(child.child_legal_actions) > 0:
                child_legal_actions = child.child_legal_actions
                child_current_player = child.current_player
                child_RNN_state = child.RNN_state
            else:
                child_legal_actions = child.child_legal_actions  # this is only used as an initialization, will be overwritten
                child_current_player = None
                child_RNN_state = None
            child_logit_priors = child.child_logit_priors
        else:
            child_legal_actions = {}  # this is only used as an initialization, will be overwritten
            child_logit_priors = None
            child_current_player = None
            child_RNN_state = None

        new_root = Root(board=self.game.board.copy(),
                        action_history=child.action_history,
                        current_player=child_current_player,
                        child_legal_actions=child_legal_actions,
                        RNN_state=child_RNN_state,
                        child_logit_priors=child_logit_priors)

        # del new_root.RNN_state # don't need this as child must be evaluated
        if child.is_terminal is None:
            new_root.children = child.children
            new_root.child_values = child.child_values
            new_root.child_visits = child.child_visits

            if len(child.child_legal_actions) == 0:
                del new_root.current_player

        if child.is_terminal is not None:
            del new_root.children, new_root.child_values, new_root.child_logit_priors, new_root.child_legal_actions, new_root.current_player

        new_root.visits = self.root.child_visits[child.child_id]
        self.root = new_root

    def prune_tree(self, action, create_new_root=False):
        # given the move set the root to the child that corresponds to the move played
        # then call set root as root is technically a different class from Node
        if not create_new_root:
            for child in self.root.children:
                if child is not None and np.array_equal(child.action_history[-1], action):
                    self._set_root(child)
                    return

        # this assumes that the tree was initialized with the other person's perspective
        # and calling prune_tree wants MCTS to play from the opponents perspective,
        # this also assumes that self.game is already updated with the first player's move
        # thus we create a root and expand it

        self.create_expand_root()


if __name__ == "__main__":

    # from tqdm import tqdm
    # import multiprocessing as mp
    from Gomoku.Gomoku import Gomoku, build_config, train_config

    # from TicTacToe.Tictactoe import TicTacToe, build_config, train_config

    # from Client_Server import Parallelized_Session, start_server, create_shared_memory, convert_to_single_info

    # game = Gomoku()

    # game.do_action((7, 7))
    # game.do_action((6, 7))
    # game.do_action((7, 6))
    # game.do_action((6, 6))
    # game.do_action((7, 5))
    #
    # game.do_action((6, 5))
    # print(game.board)
    # mcts1 = MCTS(game,
    #              [],
    #              None,
    #              c_puct_init=1.25,
    #              tau=0,
    #              use_dirichlet=False,
    #              fast_find_win=False)
    # move, probs = mcts1.run(iteration_limit=100000)
    # print(move, probs)
    # game.do_action((6, 5))

    # game.do_action((7, 4))
    # game.do_action((6, 4))

    # game = TicTacToe()
    # game.do_action((0, 0))
    # game.do_action((1, 0))
    # game.do_action((1, 1))
    # print(game.board)
    # mcts1 = MCTS(game,
    #              [],
    #              None,
    #              c_puct_init=2.5,
    #              tau=0,
    #              use_dirichlet=True,
    #              fast_find_win=False)
    # move, probs = mcts1.run(iteration_limit=100000)
    # print(move, probs)
    #

    max_shape, opt_shape = 12, 12
    providers = [
        ('TensorrtExecutionProvider', {
            # "trt_engine_cache_enable": True,
            # "trt_dump_ep_context_model": True,
            # "trt_builder_optimization_level": 5,
            # "trt_auxiliary_streams": 0,
            # "trt_ep_context_file_path": "Gomoku/Cache/",
            #
            # "trt_profile_min_shapes": f"inputs:1x15x15,input_state:{num_layers}x2x1x{embed_size},input_state_matrix:{num_layers}x1x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",
            # "trt_profile_max_shapes": f"inputs:{max_shape}x15x15,input_state:{num_layers}x2x{max_shape}x{embed_size},input_state_matrix:{num_layers}x{max_shape}x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",
            # "trt_profile_opt_shapes": f"inputs:{opt_shape}x15x15,input_state:{num_layers}x2x{opt_shape}x{embed_size},input_state_matrix:{num_layers}x{opt_shape}x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",
        }),
        # 'CUDAExecutionProvider',
        'CPUExecutionProvider'
    ]
    # sess_options = rt.SessionOptions()
    # sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

    batched_inputs_feed_info = {"inputs": [[-1, 15, 15], np.float32]}

    batched_outputs_feed_info = {"policy": [-1, 225],
                                 "value": [-1, 1]}

    # sess_options = rt.SessionOptions()
    # shms = create_shared_memory(batched_inputs_feed_info, batched_outputs_feed_info, num_workers=1)
    # No need for dtype for outputs info
    # server = mp.Process(target=start_server, args=(batched_inputs_feed_info,
    #                                    batched_outputs_feed_info,
    #                                    shms,
    #                                    providers,
    #                                    sess_options,
    #                                    "Gomoku/Grok_Zero_Train/0/model.onnx"))
    # server.start()
    # session = Parallelized_Session(0,
    #                                shms[0],
    #                                convert_to_single_info(batched_inputs_feed_info),
    #                                convert_to_single_info(batched_outputs_feed_info))

    # session = rt.InferenceSession("Gomoku/Cache/model_ctx.onnx",
    #                               # sess_options=sess_options,
    #                               providers=providers)

    # sess_options.intra_op_num_threads = 2
    # sess_options.inter_op_num_threads = 1
    # session = rt.InferenceSession("TicTacToe/Grok_Zero_Train/4/model.onnx", providers=providers)
    session = rt.InferenceSession("Gomoku/Grok_Zero_Train/4/TRT_cache/model_ctx.onnx", providers=providers)
    # session = rt.InferenceSession("Gomoku/Grok_Zero_Train/4/model.onnx", providers=providers)

    winners = [0, 0, 0]
    for game_id in range(1):
        game = Gomoku()
        game.do_action((7, 7))
        game.do_action((6, 7))
        game.do_action((7, 6))
        game.do_action((6, 6))
        game.do_action((7, 5))
        # game.do_action((6, 5))

        # game = TicTacToe()
        # game.do_action((1, 1))
        # game.do_action((0, 0))
        # game.do_action((1, 0))
        # game.do_action((1, 2))
        # game.do_action((0, 2))
        # game.do_action((2, 0))
        #
        # game.do_action((2, 2))
        # game.do_action((0, 1))

        mcts1 = MCTS_Gumbel(game,
                            # None,
                            session,
                            None,
                            m=225,
                            c_scale=1.0,
                            c_visit = 50.0,
                            fast_find_win=False)
        # mcts2 = MCTS_Gumbel(game,
        #              None,
        #              # session,
        #              None,
        #              m = 9,
        #              fast_find_win=False)

        current_move_num = 0
        winner = -2
        print(game.board)
        while winner == -2:

            if game.get_next_player() == 1:
                move, probs = mcts1.run(20000, use_bar=True)
            else:
                # move, probs = mcts2.run(2, use_bar=False)
                move = game.input_action()
                probs = []
            # legal_actions = game.get_legal_actions()
            # index = np.random.choice(np.arange(len(legal_actions)), 1)[0]
            # move = legal_actions[index]

            game.do_action(move)
            print(game.board)
            print(move, probs)
            current_move_num += 1
            winner = game.check_win()
            if winner != -2:
                # print(winner)
                winners[winner + 1] += 1
            if winner == -2:
                mcts1.prune_tree(move)
                # mcts2.prune_tree(move)
        # raise ValueError
    print(winners)

    # print(game.board)
    # move1, probs1 = mcts1.run(100000, use_bar=False)
    # move2, probs2 = mcts2.run(100000, use_bar=False)
    #
    # print(move1, move2)
    # print(probs1)
    # print(probs2)

    # winners = [0, 0, 0]
    # for _ in range(100):
    #     game = TicTacToe()
    #     mcts1 = MCTS(game,
    #                 RNN_state,
    #                 None,
    #                 c_puct_init=2.5,
    #                 tau=0.0,
    #                 use_dirichlet=True,
    #                 fast_find_win=False)
    #
    #     mcts2 = MCTS(game,
    #                 RNN_state,
    #                 None,
    #                 c_puct_init=2.5,
    #                 tau=0.0,
    #                 use_dirichlet=True,
    #                 fast_find_win=False)
    #     print(game.board)
    #     winner = -2
    #     while winner == -2:
    #         # if game.get_next_player() == -1:
    #         move, probs = mcts1.run(iteration_limit=1000, time_limit=None, use_bar=False)
    #         # else:
    #         #     move, probs = mcts2.run(iteration_limit=1000, time_limit=None, use_bar=False)
    #         game.do_action(move)
    #         print(game.board)
    #         print(move)
    #         print()
    #         # print(probs)
    #         winner = game.check_win()
    #         assert winner == game.check_win_MCTS(game.board, np.array(game.action_history), -game.get_next_player())
    #         if winner == -2:
    #             mcts1.prune_tree(move)
    #             # mcts2.prune_tree(move)
    #     print(winner)
    #     winners[winner + 1] += 1
    # print(winners)

    # print(game.board)
    # winner = -2
    # while winner == -2:
    #     if game.get_next_player() == -1:
    #         move = game.input_action()
    #         print("You played", move)
    #         # mcts = MCTS(game,
    #         #             build_config,
    #         #             session,
    #         #             c_puct_init=2.5,
    #         #             tau=0.01,
    #         #             use_dirichlet=True,
    #         #             fast_find_win=True)
    #         # move, probs = mcts.run(iteration_limit=10000, time_limit=None)
    #         game.do_action(move)
    #         print(game.board)
    #         mcts.prune_tree(move)
    #         winner = game.check_win()
    #     else:
    #         # mcts = MCTS(game,
    #         #             build_config,
    #         #             session,
    #         #             c_puct_init=2.5,
    #         #             tau=0.01,
    #         #             use_dirichlet=True,
    #         #             fast_find_win=True)
    #         move, probs = mcts.run(iteration_limit=50000, time_limit=None)
    #         game.do_action(move)
    #         # print("AI played", move)
    #         # print(probs)
    #         print(game.board)
    #
    #         mcts.prune_tree(move)
    #         winner = game.check_win()
    # print("player", winner, "won")
    # server.terminate()
