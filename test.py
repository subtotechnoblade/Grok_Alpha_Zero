# feel free to comment everything out and test your own code if you'd like
# Brian's test section
# import sys
# import time
# import numpy as np
# from numba import njit

# def calc_A(h, t):
#     return -6 * (t ** 4) + h * (t ** 3) + 2 * (t ** 2) +t
# def calc_A2(h, t):
#     return -6 * t ** 4 + h * t ** 3 + 2 * t ** 2 +t
# if __name__ == "__main__":
#     h = int(input())
#     M = int(input())
#     for T in range(1, M + 1):
#         assert calc_A(h, T) == calc_A2(h, T)
#         if calc_A(h, T) <= 0:
#             print(T)
#             break
#     else:
#         print("Failed")
#


if __name__ == "__main__":
    print(7 // 4)

