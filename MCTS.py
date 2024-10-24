import numpy as np

class Node1:
    __slots__ = "state", "move", "move_history", "parent", "children", "visits", "value", "prob_prior"
    def __init__(self, state : np.array, move : np.array, move_history : list, parent=None):
        self.state = state
        self.move = move
        self.move_history = move_history
        self.parent = parent
        self.children = []

        self.visits = 0
        self.value = 0
        self.prob_prior = 0

class Node2:
    __slots__ = "state", "move", "move_history", "parent", "children"
    def __init__(self, state : np.array, move : np.array, move_history : list, parent=None):
        self.state = state
        self.move = move
        self.move_history = move_history
        self.parent = parent
        self.children = []
class MCTS:
    def __init__(self,):
        pass

    def PUCT_select(self, node):
        pass

    def expand(self, node):
        pass

    def back_prop(self, node):
        pass

if __name__ == "__main__":
    node = Node2(np.zeros((3, 3)), (1, 1), [])
