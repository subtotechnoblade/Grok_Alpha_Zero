import h5py as h5
import numpy as np
from glob import glob

class Pad_Dataset:
    def __init__(self, folder_path, num_previous_generations):
        current_generation = max([int(path.split("/")[-1]) for path in glob(f"{folder_path}/*")])
        self.file_paths = [path + "/Self_Play_Data.h5" for path in glob(f"{folder_path}/*")][max(current_generation - num_previous_generations, 0): current_generation + 1]

        self.max_actions = 0 # get the max moves for the previous datasets up to
        for path in self.file_paths:
            with h5.File(path) as file:
                if file["max_actions"][0] > self.max_actions:
                    self.max_actions = file["max_actions"][0]



    def pad_dataset(self):
        print("Padding the datasets!")
        for file_path in self.file_paths:
            with h5.File(file_path, mode="a", libver="latest") as file:
                max_actions = file["max_actions"][0]

                if max_actions > self.max_actions:
                    raise ValueError("self.max moves is smaller that this dataset's max moves.")

                samples = (len(file.keys()) - 1) // 3

                for game_id in range(samples):
                    if len(file[f"values_{game_id}"]) == self.max_actions:
                        # if the game satisfies self.max_actions the don't do anything
                        continue
                    game_len = len(file[f"values_{game_id}"])

                    # resize which expands
                    # or retracts the array which deletes padding if self.max_actions is smaller and vice versa
                    file[f"boards_{game_id}"].resize(self.max_actions, axis=0)
                    file[f"policies_{game_id}"].resize(self.max_actions, axis=0)
                    file[f"values_{game_id}"].resize(self.max_actions, axis=0)

                    if game_len < self.max_actions:
                        # pad the boards with -2.0
                        file[f"boards_{game_id}"][game_len:] = np.array(-2, dtype=file[f"boards_0"][0].dtype) # Basic broadcasting

                        # pad the policies
                        file[f"policies_{game_id}"][game_len:] = -2.0 # Note that these are float because policy is of type float

                        # pad the values
                        file[f"values_{game_id}"][game_len:] = -2.0

if __name__ == "__main__":
    pad = Pad_Dataset("Gomoku/Grok_Zero_Train/", 1)
    pad.pad_dataset()
    with h5.File("Gomoku/Grok_Zero_Train/0/Self_Play_Data.h5", "r") as file:
        print(len(file["values_0"]))

    # with h5.File("Gomoku/Grok_Zero_Train/1/Self_Play_Data.h5", "r") as file:
    #     print(len(file["values_0"]))




