# feel free to comment everything out and test your own code if you'd like
# Brian's test section

import h5py as h5
import numpy as np
import multiprocessing as mp
from numba import njit, prange
from tqdm import  tqdm


if __name__ == "__main__":
    def check_for_duplicated_unhashable(input_list):
        seen = 0

        # O(n^2) time complexity which is very slow
        # Should be fine for small lists of size under 1000
        for item in input_list:
            for check_item in input_list:
                if item == check_item:
                    seen += 1
                    if seen == 2:
                        return False
            seen = 0
        return True
    arr1 = np.array([[1, 2], [2, 1]])
    arr2 = np.array([1, 2])
    print(arr1 == arr2)
    print(np.where(np.all(arr1 == arr2, axis=1)))
    class Foo:
        def bar(self):
            pass
    f= Foo()
    print(f.grr())

    # def task(lock, i):
    #
    #     with lock, h5.File("test.h5", "r+") as f:
    #         f["0"].resize((8,))
    #         f["0"][7] = i + 1
    #
    #     print(f"Done{i}")
    #
    #
    # with h5.File("test.h5", "w", libver="latest") as f:
    #     f.create_dataset(f"{0}", maxshape=(None,), dtype=np.int32, data=np.zeros(6))
    # lock = mp.Lock()
    # jobs = []
    # for job_id in range(6):
    #     p = mp.Process(target=task, args=(lock, job_id,))
    #     p.start()
    #     jobs.append(p)
    #
    # for p in jobs:
    #     p.join()
    #
    # with h5.File("test.h5", "r+") as f:
    #     print(np.array(f["0"]))





