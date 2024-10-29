# feel free to comment everything out and test your own code if you'd like
# Brian's test section
import sys
import numpy as np
from numba import njit
from collections import deque
# class Foo:
#     @staticmethod
#     def bar(x):
#         return x
#
#     def bar(self, x):
#         return x
#
# def bar(x):
#     return x

class Node:
    __slots__ = "bar"

    def __init__(self, board):
        self.bar = board.copy()

    def foo(self):
        self.bar[1] = 5

    # return actions, arr2

if __name__ == "__main__":
    import time
    print(sys.getsizeof(None))
    # board = np.array([1 ,2 ,3])
    # node = Node(board)
    # node.foo()
    # print(board)
    # print(node.bar)



