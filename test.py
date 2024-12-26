# feel free to comment everything out and test your own code if you'd like
# Brian's test section

import h5py as h5
import numpy as np
import multiprocessing as mp
from numba import njit, prange
from tqdm import tqdm


if __name__ == "__main__":

    def task(lock, i):

        with lock, h5.File("test.h5", "r+") as f:
            f.create_dataset(f"{i}", maxshape=(None,), dtype=np.int32, data=np.zeros(1))
            f[f"{i}"].resize((8,))
            f[f"{i}"][7] = i + 1

        print(f"Done{i}")


    with h5.File("test.h5", "w", libver="latest") as f:
        f.create_dataset(f"{0}", maxshape=(None,), dtype=np.int32, data=np.zeros(6))
    lock = mp.Lock()
    jobs = []
    for job_id in range(1, 7):
        p = mp.Process(target=task, args=(lock, job_id,))
        p.start()
        jobs.append(p)

    for p in jobs:
        p.join()

    with h5.File("test.h5", "r+") as f:
        print(np.array(f["0"]))
        print(f.keys())
        print(np.array(f["1"]))





