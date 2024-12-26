# def _check_row(row):
#     if row[0] != 0 and row[0] == row[1] == row[2]:
#         return row[0]
#     return None
import numpy as np
if __name__ == "__main__":
    board = np.zeros((3, 3))
    # test_row = [1, 1, 1]
    # print (_check_row(test_row))
    # for row in board:
    #     for value in row:
    #         if value == 0:
    #             break
    flattened_board = board.reshape((9,))
    for value in flattened_board:
        if value == 0:
            break
    else:
        print("Draw")