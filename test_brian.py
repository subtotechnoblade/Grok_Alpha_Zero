import numpy as np

if __name__ == "__main__":
    x = np.zeros((3, 3))

    y = x.copy()
    y[0][0] = -1
    print(y)
    print(x)