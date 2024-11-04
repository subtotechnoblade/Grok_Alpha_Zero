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
    N_forwards, N_backwards, B_forwards, B_backwards, total_steps = [int(input()) for _ in range(5)]

    def calc_delta_d(forwards, backwards, total_steps):
        steps = 0
        delta_d = 0
        step_direction = 1
        while True:
            if step_direction == 1:
                if steps + forwards < total_steps:
                    steps += forwards
                    delta_d += forwards
                else:
                    delta_d += total_steps - steps
                    break

            else:
                if steps + backwards < total_steps:
                    steps += backwards
                    delta_d -= backwards
                else:
                    delta_d += total_steps - steps
                    break
            step_direction *= -1
        return delta_d
    N_delta_d = calc_delta_d(N_forwards, N_backwards, total_steps)
    B_delta_d = calc_delta_d(B_forwards, B_backwards, total_steps)
    print("Nikky" if N_delta_d > B_delta_d else ("Byron" if N_delta_d > B_delta_d else "Tied"))


# if x is odd there must be an
# p = [p1 * p2 * ... * pt * n1 * ... * nk]

# because the p can't be decomposed anymore
# the definition of a prime means that it can't be decomposed anymore
# thus

