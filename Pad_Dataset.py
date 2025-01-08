import numpy as np
import h5py as h5

class Pad_Dataset:
    def __init__(self, file_path):
        self.file_path = file_path
    def pad_dataset(self):
        print("Padding the dataset!")
        with h5.File(self.file_path, mode="a", libver="latest") as file:
            samples = (len(file.keys()) - 1) // 3
            max_moves = file["max_moves"][0]

            # use zeros for board
            board_padding = np.expand_dims(np.zeros_like(file[f"boards_0"][0], dtype=file[f"boards_0"][0].dtype), 0 )

            # use -2 as the padding value
            policy_padding = np.expand_dims(np.ones_like(file[f"policies_0"][0], dtype=np.float32), 0) * -2.0
            value_padding = np.expand_dims(np.ones_like(file[f"values_0"][0], dtype=np.float32), 0) * -2.0

            for game_id in range(samples):
                if len(file[f"values_{game_id}"]) == max_moves:
                    # if the game is the longest one, we don't pad it
                    continue
                game_len = len(file[f"values_{game_id}"])


                # pad the boards
                file[f"boards_{game_id}"].resize(max_moves, axis=0)
                file[f"boards_{game_id}"][game_len:] = np.repeat(board_padding, max_moves - game_len, axis=0)

                # pad the policies
                file[f"policies_{game_id}"].resize(max_moves, axis=0)
                file[f"policies_{game_id}"][game_len:] = np.repeat(policy_padding, max_moves - game_len, axis=0)

                # pad the values
                file[f"values_{game_id}"].resize(max_moves, axis=0)
                file[f"values_{game_id}"][game_len:] = np.repeat(value_padding, max_moves - game_len, axis=0)

if __name__ == "__main__":
    pad = Pad_Dataset("Gomoku/Grok_Zero_Train/0/Self_Play_Data.h5")
    # pad.pad_dataset()
    with h5.File("Gomoku/Grok_Zero_Train/0/Self_Play_Data.h5", "r") as file:
        print(file["values_0"][:].reshape(-1))
        print(file["values_1"][:].reshape(-1))

        print(len(file["values_0"]))
        values = file["values_0"][:].reshape(-1)
        values = values[values != -2]
        print(values)
        print(len(values))
        print(len(file["values_1"]))



