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
    __slots__ = "foo", "bar"

    def __init__(self):
        self.bar = []
@njit(cache=True)
def numba_func(arr1):
    return np.argwhere(arr1)

    # return actions, arr2

if __name__ == "__main__":
    import time
    s = time.time()
    print(numba_func(np.array([[1, 2, 3], [1, 2, 3]])))

