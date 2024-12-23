# feel free to comment everything out and test your own code if you'd like
# Brian's test section

import h5py as h5
import numpy as np
import multiprocessing as mp
from numba import njit, prange
from tqdm import  tqdm


if __name__ == "__main__":
    from numba import njit
    ttuple = lambda x: tuple(tuple(thing) for thing in x)
    # x = set()
    # x.add(ttuple([[1, 2]],))
    # print(x)
    # raise ValueError
    # board = np.array([[1, 2, 3, 3],
    #          [4, 5, 6, 5],
    #          [7, 8, 9, 1],
    #                   [1, 2, 3, 5]])
    board = np.arange(9).reshape((3, 3))

    print(board)
    up_board = np.flipud(board)
    print(up_board)
    lr_board = np.fliplr(board)
    print(lr_board)
    print("\n")


    augmented_boards = {ttuple(board), ttuple(up_board), ttuple(lr_board)}
    for k in range(1, 4):
        rot_board = ttuple(np.rot90(board, k))
        augmented_boards.add(ttuple(rot_board))
        if k == 1:
            augmented_boards.add(ttuple(np.flipud(rot_board)))
            augmented_boards.add(ttuple(np.fliplr(rot_board)))


    for k in range(1, 4):
        print("K is:", k)
        rot_board = np.rot90(board, k)
        print(rot_board)
        if ttuple(rot_board) in augmented_boards:
            print(f"{k} rot is a repeat")
            # print(rot_board)
            # print(augmented_boards)
        augmented_boards.add(ttuple(rot_board))

        flipped_up_board = np.flipud(rot_board)
        if ttuple(flipped_up_board) in augmented_boards:
            print(f"{k} ud is a repeat")
            # print(flipped_up_board)
            # print(augmented_boards)
        augmented_boards.add(ttuple(flipped_up_board))

        flipped_lr_board = np.fliplr(rot_board)
        if ttuple(flipped_lr_board) in augmented_boards:
            print(f"{k} lr is a repeat")
            # print(flipped_lr_board)
            # print(augmented_boards)
        augmented_boards.add(ttuple(flipped_lr_board))
        print(flipped_lr_board)
        print(flipped_up_board)
        # print("\n")
    # raise ValueError

    # note base don the experiements
    # we only need the original board + flipup + fliplr
    # and rot_board (k = 1) + flipup + fliplr
    # rot_board (k = 2)
    # rot_board (k = 3)





















    # def check_for_duplicated_unhashable(input_list):
    #     seen = 0
    #
    #     # O(n^2) time complexity which is very slow
    #     # Should be fine for small lists of size under 1000
    #     for item in input_list:
    #         for check_item in input_list:
    #             if item == check_item:
    #                 seen += 1
    #                 if seen == 2:
    #                     return False
    #         seen = 0
    #     return True
    # arr1 = np.array([[1, 2], [2, 1]])
    # arr2 = np.array([1, 2])
    # print(arr1 == arr2)
    # print(np.where(np.all(arr1 == arr2, axis=1)))
    # class Foo:
    #     def bar(self):
    #         pass
    # f= Foo()
    # print(f.grr())

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





