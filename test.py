# feel free to comment everything out and test your own code if you'd like
# Brian's test section

import h5py as h5
import numpy as np
import multiprocessing as mp
from numba import njit, prange
from tqdm import tqdm


if __name__ == "__main__":
    import tensorflow as tf
    y_true = [[0, 1, 0, 0], [1, 0, 0, 0]]
    y_pred = [[-18.6, 0.51, 2.94, -12.8], [-18.6, 0.51, 2.94, -12.8]]
    # bce = tf.keras.losses.BinaryCrossentropy(from_logits=True, )
    # print(bce(y_true, y_pred))

    y_true  = tf.convert_to_tensor(y_true)
    print(y_true[:, 0])
    # print(tf.cast(y_true == 0, tf.float32))
    raise ValueError

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





