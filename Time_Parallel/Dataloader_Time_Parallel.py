import h5py as h5
import numpy as np
from glob import glob

import tensorflow as tf


class Create_Train_Test_Split_Indexes:
    def __init__(self,
                 parent_path,
                 num_previous_generations,
                 train_percent=1.0,
                 train_decay=0.75,
                 test_percent=0.1,
                 test_decay=0.75):
        self.parent_path = parent_path
        np.random.seed()

        self.generation = max([int(path.split("/")[-1]) for path in glob(self.parent_path + "/*")])

        self.files = [path + "/Self_Play_Data.h5" for path in glob(self.parent_path + "/*")][-num_previous_generations - 1:]
        self.files.reverse()

        # files are from latest to first generation, [n, ..., 0]
        # Note that if there is no previous generation, or if we expected more, this will still work

        self.train_percent = train_percent
        self.train_decay = train_decay
        self.test_percent = test_percent
        self.test_decay = test_decay

        self.indexes = [None] * len(self.files)
        for i, path in enumerate(self.files):
            file = h5.File(path, mode="r")
            samples = (len(file.keys()) - 1) // 3
            if samples < 2:
                raise ValueError("A generation's dataset has to contain at least 2 games! 1 for train and 1 for split")
            self.indexes[i] = set(range(samples))

    def split(self):
        train_percent = self.train_percent
        test_percent = self.test_percent
        train_indexes, test_indexes = [None] * len(self.files), [None] * len(self.files)

        for i, indexes in enumerate(self.indexes):
            random_test_indexes = np.random.choice(list(indexes), max(int(len(indexes) * test_percent), 1), replace=False) # same as expand dims 1
            test_indexes[i] = np.concatenate((np.ones((len(random_test_indexes), 1), dtype=np.int32) * (self.generation - i),
                                              random_test_indexes.reshape((-1, 1))), axis=-1, dtype=np.int32)

            train_indexes_generation = np.array(list(indexes - set(random_test_indexes)), dtype=np.int32)

            train_indexes_generation = np.random.choice(train_indexes_generation, int(len(train_indexes_generation) * train_percent), replace=False)
            train_indexes_generation = train_indexes_generation.reshape((-1, 1)) # same as expand dims 1
            train_indexes[i] = np.concatenate((np.ones((len(train_indexes_generation), 1), dtype=np.int32) * (self.generation - i), train_indexes_generation), axis=1)

            test_percent *= self.test_decay
            train_percent *= self.train_decay

        #returns the latest generation to the earliest
        return np.concatenate((*train_indexes,), dtype=np.int32), np.concatenate((*test_indexes,), dtype=np.int32)

class Dataloader(tf.keras.utils.PyDataset):
    def __init__(self,
                 folder_path,
                 indexes,
                 batch_size,
                 **kwargs):
        super().__init__(**kwargs)
        self.folder_path = folder_path
        self.indexes = indexes
        self.batch_size = batch_size
        num_previous_generations = self.indexes[:, 0].max()
        self.files = [path + "/Self_Play_Data.h5" for path in glob(self.folder_path + "/*")][-num_previous_generations - 1:]

        self.datasets = [None] * len(self.files)
        for i, path in enumerate(self.files):
            self.datasets[i] = h5.File(path, mode="r")

        self.on_epoch_end()

    def __len__(self):
        return len(self.batch_indexes)

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

        num_samples = len(self.indexes)
        if num_samples % self.batch_size == 0:
            self.batch_indexes = [*self.indexes.reshape((-1, self.batch_size, 2))]
        elif num_samples < self.batch_size:
            self.batch_indexes = [*self.indexes.reshape((-1, num_samples, 2))]
        else:
            remainder = num_samples % self.batch_size
            self.batch_indexes = [*self.indexes[:-remainder].reshape((-1, self.batch_size, 2)), self.indexes[-remainder:]]


    def __getitem__(self, idx):
        batch_size = len(self.batch_indexes[idx])
        boards = [None] * batch_size
        policies = [None] * batch_size
        values = [None] * batch_size
        for i, (dataset_id, index) in enumerate(self.batch_indexes[idx]):
            file = self.datasets[dataset_id]
            boards[i] = file[f"boards_{index}"][:]
            policies[i] = file[f"policies_{index}"][:]
            values[i] = file[f"values_{index}"][:]
        return np.array(boards, np.float32), (np.array(policies, np.float32), np.array(values, np.float32))

def Create_Dataset(folder_path,
                   num_previous_generations,
                   train_batch_size,
                   test_batch_size=None,
                   train_percent=1.0,
                   train_decay=0.75,
                   test_percent=0.1,
                   test_decay=0.75):
    train_indexes, test_indexes = Create_Train_Test_Split_Indexes(folder_path,
                                                                  num_previous_generations,
                                                                  train_percent,
                                                                  train_decay,
                                                                  test_percent,
                                                                  test_decay).split()

    train_dataloader = Dataloader(folder_path, train_indexes, train_batch_size)
    test_dataloader = Dataloader(folder_path, test_indexes, train_batch_size if test_batch_size is None else test_batch_size)
    return train_dataloader, test_dataloader




if __name__ == "__main__":
    from Gomoku.Gomoku import Gomoku
    np.set_printoptions(threshold=np.inf)
    game = Gomoku()
    split = Create_Train_Test_Split_Indexes("../Gomoku/Grok_Zero_Train/", 3)
    train_indexes, test_indexes = split.split()
    print(train_indexes.shape, test_indexes.shape)

    dataloader = Dataloader("../Gomoku/Grok_Zero_Train/",
                            train_indexes,
                            batch_size=3)
    boards, (policy, value) = dataloader[len(dataloader) - 1]
    print(boards.shape)
    # print(boards[0])



