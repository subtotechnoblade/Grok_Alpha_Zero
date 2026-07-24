import numpy as np
arr = np.array([[27, 9],
       [32, 4],
       [15, 21],
       [5, 31],
       [1, 35],
       ])

arr = 1/arr
arr = np.sum(arr, axis=-1)
arr = 1/arr
print(arr)
print(np.sum(arr)/len(arr))

