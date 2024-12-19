import numpy as np

if __name__ == "__main__":
    x = np.array([0.5, 0.2, 0.3])
    y = x ** (1 / 0.1)
    print(x)
    print(y)
    print(y / np.sum(y))