# feel free to comment everything out and test your own code if you'd like
# Brian's test section

import h5py as h5
import numpy as np
import multiprocessing as mp
from numba import njit, prange, types
from tqdm import tqdm


# def encode_protocol(arrs):
#     # for arr in arrs:
#     #     if arr.dtype != np.float32:
#     #         raise ValueError(f"Arr dtype must be np.float32 was {arr.dtype}")
#     encoded_data = [1.0, len(arrs)]
#     for arr in arrs:
#         encoded_data += [arr.ndim, *arr.shape, *arr.flatten()]
#     return np.array(encoded_data, dtype=np.float32)
#     # encoded_data = np.array(encoded_data, dtype=np.float32).flatten()
#
#     # the protocol is [synchronization bit (0 for no data, 1 for data),
#     # num of arrays,
#
#     # len of arr0.shape,
#     # arr0.shape,
#     # arr0.flatten()
#
#     # process above is repeated to num of arrays amount of times
#     # ]
#
#     # return np.concatenate((np.array([1.0, len(arrs)]), encoded_data))

# @njit(cache=True)
# def decode(encoded_arr):
#     # the 0th index used for knowing if there is a message sent thus it is not used
#     num_arrs = int(encoded_arr[1])
#
#     arrs = []
#     arr_shapes = []
#
#     start_index = 2
#     for i in range(num_arrs):
#         arr_shape_len = int(encoded_arr[start_index])
#
#         arr_shape = encoded_arr[start_index + 1: start_index + 1 + arr_shape_len].astype(np.int32)
#
#         arr_len = np.prod(arr_shape)
#
#         arr_shapes.append(arr_shape)
#         arrs.append(encoded_arr[start_index + 1 + arr_shape_len: start_index + 1 + arr_shape_len + arr_len])
#
#         start_index += arr_shape_len + arr_len + 1
#     return arrs, arr_shapes
#
# def decode_protocol(encoded_arr):
#     unshaped_arrs_shape = decode(encoded_arr)
#     arrs = [0] * len(unshaped_arrs_shape[0])
#
#     for i, (vector, shape) in enumerate(zip(*unshaped_arrs_shape)):
#         arrs[i] = vector.reshape(shape)
#     return arrs



if __name__ == "__main__":
    import time
    import multiprocessing as mp















    # from multiprocessing.shared_memory import SharedMemory
    # d_size = 4 * 2
    # arr = np.array([0.0, 0.0], dtype=np.float32)
    # shm = SharedMemory(create=True, size=d_size)
    # shm_arr = np.ndarray(shape=(shm.size // 4,), dtype=np.float32, buffer=shm.buf)
    # def task(shm):
    #     time.sleep(1)
    #     shm_arr = np.ndarray(shape=(shm.size // 4,), dtype=np.float32, buffer=shm.buf)
    #     shm_arr[:] = 1.0
    #
    # # p = mp.Process(target=task, args=(shm,))
    # # p.start()
    # print(shm_arr)
    # # while True:
    # #     if shm_arr[0] == 1.0:
    # #         print(shm_arr)
    # #         print("Done")
    # #         break
    # # p.join()
    #
    # shm.unlink()


    # def task(lock, i):
    #
    #     with lock, h5.File("test.h5", "r+") as f:
    #         f.create_dataset(f"{i}", maxshape=(None,), dtype=np.int32, data=np.zeros(1))
    #         f[f"{i}"].resize((8,))
    #         f[f"{i}"][7] = i + 1
    #
    #     print(f"Done{i}")
    #
    #
    # with h5.File("test.h5", "w", libver="latest") as f:
    #     f.create_dataset(f"{0}", maxshape=(None,), dtype=np.int32, data=np.zeros(1,))
    #     if f["0"][0] == 0:
    #         f["0"][0] = 1
    #     print(f["0"][0])
    #     print(isinstance(5 // 3, int))

    # lock = mp.Lock()
    # jobs = []
    # for job_id in range(1, 7):
    #     p = mp.Process(target=task, args=(lock, job_id,))
    #     p.start()
    #     jobs.append(p)
    #
    # for p in jobs:
    #     p.join()
    #
    # with h5.File("test.h5", "r+") as f:
    #     print(np.array(f["0"]))
    #     print(f.keys())
    #     print(np.array(f["1"]))





