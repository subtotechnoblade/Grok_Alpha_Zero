# feel free to comment everything out and test your own code if you'd like
# Brian's test section
import sys
import numpy as np
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


if __name__ == "__main__":
    import time
    foo = Node()
    bar = Node()

    del foo.bar
    s = time.time()
    for _ in range(10000):
        hasattr(foo, "foo")
    print(time.time() - s)



    s = time.time()
    for _ in range(10000):
        if bar.bar:
            pass
    del bar.bar
    print(time.time() - s)
    # del foo.bar
    # del bar.foo
    # print(sys.getsizeof(foo))
    # print(sys.getsizeof(bar))
    #
    # # Total size including NumPy arrays for foo
    print(sys.getsizeof(foo))
    print(sys.getsizeof(bar) + sys.getsizeof(bar.bar))

    # print(sys.getsizeof(Foo.bar))
    # print(sys.getsizeof(bar))
