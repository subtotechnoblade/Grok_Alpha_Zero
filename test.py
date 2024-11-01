# feel free to comment everything out and test your own code if you'd like
# Brian's test section
import sys
import time
import numpy as np
from numba import njit

x = np.array([1, 2, 3])
y = x
del x
y[0] = 2
if __name__ == "__main__":
    print(y)





