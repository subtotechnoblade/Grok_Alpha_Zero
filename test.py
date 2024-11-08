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

import h5py as h5
import numpy as np
import multiprocessing as mp

if __name__ == "__main__":

    def task(lock, i):

        with lock, h5.File("test.h5", "r+") as f:
            f["0"][i] = i + 1

        print(f"Done{i}")


    with h5.File("test.h5", "w", libver="latest") as f:
        f.create_dataset(f"{0}", maxshape=(None,), dtype=np.int32, data=np.zeros(6))
    lock = mp.Lock()
    jobs = []
    for job_id in range(6):
        p = mp.Process(target=task, args=(lock, job_id,))
        p.start()
        jobs.append(p)

    for p in jobs:
        p.join()

    with h5.File("test.h5", "r+") as f:
        print(np.array(f["0"]))






