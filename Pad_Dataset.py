import h5py as h5
import numpy as np
from glob import glob

class Pad_Dataset:
    def __init__(self, folder_path, previous_generations):
        self.file_paths = [path + "/Self_Play_Data.h5" for path in glob(f"{folder_path}/*")][:previous_generations + 1]

        self.max_moves = 0 # get the max moves for the previous datasets up to
        for path in self.file_paths:
            with h5.File(path) as file:
                if file["max_moves"][0] > self.max_moves:
                    self.max_moves = file["max_moves"][0]



    def pad_dataset(self):
        print("Padding the datasets!")
        for file_path in self.file_paths:
            with h5.File(file_path, mode="a", libver="latest") as file:
                max_moves = file["max_moves"][0]

                if max_moves > self.max_moves:
                    raise ValueError("self.max moves is smaller that this dataset's max moves.")

                samples = (len(file.keys()) - 1) // 3

                # use zeros for board
                # board_padding = np.expand_dims(np.zeros_like(file[f"boards_0"][0], dtype=file[f"boards_0"][0].dtype), 0 )
                #
                # # use -2 as the padding value
                # policy_padding = np.expand_dims(np.ones_like(file[f"policies_0"][0], dtype=np.float32), 0) * -2.0
                # value_padding = np.expand_dims(np.ones_like(file[f"values_0"][0], dtype=np.float32), 0) * -2.0

                for game_id in range(samples):
                    if len(file[f"values_{game_id}"]) == self.max_moves:
                        # if the game satisfies self.max_moves the don't do anything
                        continue
                    game_len = len(file[f"values_{game_id}"])

                    # resize which expands
                    # or retracts the array which deletes padding if self.max_moves is smaller and vice versa
                    file[f"boards_{game_id}"].resize(self.max_moves, axis=0)
                    file[f"policies_{game_id}"].resize(self.max_moves, axis=0)
                    file[f"values_{game_id}"].resize(self.max_moves, axis=0)

                    if game_len < self.max_moves:
                        # pad the boards
                        # file[f"boards_{game_id}"][game_len:] = np.repeat(board_padding, self.max_moves - game_len, axis=0)
                        file[f"boards_{game_id}"][game_len:] = np.array(-2, dtype=file[f"boards_0"][0].dtype)
                        # pad the policies
                        # file[f"policies_{game_id}"][game_len:] = np.repeat(policy_padding, self.max_moves - game_len, axis=0)
                        file[f"policies_{game_id}"][game_len:] = -2.0

                        # pad the values
                        file[f"values_{game_id}"][game_len:] = -2.0

if __name__ == "__main__":
    pad = Pad_Dataset("Gomoku/Grok_Zero_Train/", 1)
    pad.pad_dataset()
    with h5.File("Gomoku/Grok_Zero_Train/0/Self_Play_Data.h5", "r") as file:
        print(len(file["values_0"]))

    # with h5.File("Gomoku/Grok_Zero_Train/1/Self_Play_Data.h5", "r") as file:
    #     print(len(file["values_0"]))




