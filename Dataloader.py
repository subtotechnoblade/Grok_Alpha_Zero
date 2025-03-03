import h5py as h5
import numpy as np
from glob import glob
from tqdm import tqdm

import tensorflow as tf

class Create_Train_Test_Split:
    def __init__(self,
                 parent_path,
                 num_previous_generations,
                 train_percent=1.0,
                 train_decay=0.95,
                 test_percent=0.1,
                 test_decay=0.95):
        self.parent_path = parent_path
        np.random.seed()


        self.generation = max([int(path.split("/")[-1]) for path in glob(self.parent_path + "/*")])

        self.files = [path + "/Self_Play_Data.h5" for path in glob(self.parent_path + "/*")][-num_previous_generations - 1:]

        self.train_percent = train_percent
        self.train_decay = train_decay
        self.test_percent = test_percent
        self.test_decay = test_decay



    def split(self):
        print(f"Collecting data from {len(self.files)} generations!")
        self.train_states, self.train_policies, self.train_values = [], [], []
        self.test_states, self.test_policies, self.test_values = [], [], []

        train_percent = self.train_percent
        test_percent = self.test_percent

        for file_path in self.files:
            self.states_generation, self.policies_generation, self.values_generation = [], [], []
            with h5.File(file_path, mode="r") as file:
                num_games = (len(file.keys()) - 1) // 3

                for game_id in tqdm(range(num_games), desc="Collecting data"):
                    self.states_generation += list(file[f"boards_{game_id}"])
                    self.policies_generation += list(file[f"policies_{game_id}"])
                    self.values_generation += list(file[f"values_{game_id}"])

            self.states_generation, self.policies_generation, self.values_generation = np.ascontiguousarray(self.states_generation), np.ascontiguousarray(self.policies_generation), np.ascontiguousarray(self.values_generation)
            np.random.shuffle(self.states_generation)
            np.random.shuffle(self.policies_generation)
            np.random.shuffle(self.values_generation)

            num_actions = len(self.states_generation)
            test_indexes = np.random.choice(np.arange(num_actions), size=int(test_percent * num_actions), replace=False)
            self.test_states += list(self.states_generation[test_indexes])
            self.test_policies += list(self.policies_generation[test_indexes])
            self.test_values += list(self.values_generation[test_indexes])


            train_indexes = np.asarray(list(set(range(num_actions)) - set(test_indexes)))
            train_indexes = np.random.choice(train_indexes, size=int(train_percent * len(train_indexes)), replace=False)
            self.train_states += list(self.states_generation[train_indexes])
            self.train_policies += list(self.policies_generation[train_indexes])
            self.train_values += list(self.values_generation[train_indexes])

        return (self.train_states, self.train_policies, self.train_values), (self.test_states, self.test_policies, self.test_values)


class Dataloader(tf.keras.utils.PyDataset):
    def __init__(self, states, policies, values, batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.states = states
        self.policies = policies
        self.values = values

        self.batch_size = batch_size
        self.num_samples = len(states)

        self.indexes = np.random.permutation(self.num_samples)

        self.on_epoch_end()

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

        num_samples = len(self.indexes)
        if num_samples % self.batch_size == 0:
            self.batch_indexes = [*self.indexes.reshape((-1, self.batch_size,))]
        elif num_samples < self.batch_size:
            self.batch_indexes = [*self.indexes.reshape((-1, num_samples,))]
        else:
            remainder = num_samples % self.batch_size
            self.batch_indexes = [*self.indexes[:-remainder].reshape((-1, self.batch_size,)), self.indexes[-remainder:]]

    def __getitem__(self, idx):
        batched_indexes = self.batch_indexes[idx]
        len_batched_indexes = len(batched_indexes)
        states = np.zeros((len_batched_indexes, *self.states[0].shape), dtype=self.states[0].dtype)
        policies = np.zeros((len_batched_indexes, *self.policies[0].shape), dtype=self.policies[0].dtype)
        values = np.zeros((len_batched_indexes, *self.values[0].shape), dtype=self.values[0].dtype)

        for i, batched_index in enumerate(batched_indexes):
            states[i] = self.states[batched_index]
            policies[i] = self.policies[batched_index]
            values[i] = self.values[batched_index]
        return states, (policies, values)

def Create_Dataset(folder_path,
                   num_previous_generations,
                   train_batch_size,
                   test_batch_size=None,
                   train_percent=1.0,
                   train_decay=0.75,
                   test_percent=0.1,
                   test_decay=0.75):
    (train_states, train_policies, train_values), (test_states, test_policies, test_values) = Create_Train_Test_Split(folder_path,
                                                          num_previous_generations,
                                                          train_percent,
                                                          train_decay,
                                                          test_percent,
                                                          test_decay).split()

    train_dataset = Dataloader(train_states, train_policies, train_values, train_batch_size)
    test_dataset = Dataloader(test_states, test_policies, test_values, train_batch_size if test_batch_size is None else test_batch_size)
    return train_dataset, test_dataset

if __name__ == "__main__":
    parent_path = "Gomoku/Grok_Zero_Train/"
    split = Create_Train_Test_Split(parent_path, 1)
    (train_state, train_policies, train_values), (test_states, test_policies, test_values) = split.split()
    train_state = np.array(train_state, copy=False)

    data_loader = Dataloader(train_state, train_policies, train_values, 64)
    batched_states, (batched_policies, batched_values) = data_loader[0]
    print(batched_states.shape)
    print(batched_policies.shape)
    print(batched_values.shape)