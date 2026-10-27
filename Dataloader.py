from pathlib import Path
from warnings import warn
import h5py as h5
import numpy as np
from numba import njit

from glob import glob
from tqdm import tqdm

import tensorflow as tf

def dedup_average(states, policies, values):
    """Merge byte-identical states into ONE sample whose targets are the
    occurrence-weighted mean of the originals.

    Why (measured on connect4, see tmp/dup_audit.py): ~66% of a generation's
    samples are exact copies of another sample, and the value labels of those
    copies disagree wildly for positions the search cannot read to the end
    (std ~0.4 on a target in [-1, 1] for the first ~15 plies). MSE-SGD would
    converge to the group mean anyway - handing it the mean directly removes
    the label noise, and collapsing the copies removes the 400:1 loss-weight
    imbalance between opening and midgame states.

    Call this per split side AFTER the game-id split (never across it), on the
    POOLED window of generations: a state that repeats across generations gets
    one vote per surviving occurrence, so the train_percent recency ramp is
    preserved inside each averaged target. Deduping per generation instead
    would re-weight old generations' opinions up to equal votes - don't.

    Also returns the occurrence COUNT per unique state, to be used as the
    training sample weight. In plain SGD frequency IS the loss weight, and for
    losses linear in the target (crossentropy, MSE) the gradient of one sample
    at the mean target with weight n is IDENTICAL to the summed gradient of the
    n originals - so dedup + mean + count-weight keeps the noise reduction with
    zero distribution distortion. Dropping the count (as this first did) turned
    the empty board from 2.8% of the training mass into one sample of 0.0027%,
    a ~1000x deweight: measured across gens, the net's root argmax NEVER matched
    its own root target (agreement 0.000) - the opening prior handed to the
    search each generation was simply unfitted.
    """
    if len(states) == 0:
        return list(states), list(policies), list(values), np.ones(0, dtype=np.float32)

    states = np.asarray(states)
    policies = np.asarray(policies, dtype=np.float64)  # float64: groups reach 100s of members
    values = np.asarray(values, dtype=np.float64).reshape(len(states), -1)

    flat = states.reshape(len(states), -1)
    uniq, inverse, counts = np.unique(flat, axis=0, return_inverse=True, return_counts=True)

    policy_means = np.zeros((len(uniq), policies.shape[1]))
    value_means = np.zeros((len(uniq), values.shape[1]))
    np.add.at(policy_means, inverse, policies)
    np.add.at(value_means, inverse, values)
    policy_means /= counts[:, None]
    value_means /= counts[:, None]
    # a mean of normalized distributions is still normalized, no renorm needed

    # return lists of per-sample arrays, exactly the shape split() accumulates
    return (list(uniq.reshape(-1, *states.shape[1:])),
            list(policy_means.astype(np.float32)),
            list(value_means.astype(np.float32)),
            counts.astype(np.float32))



def split_heavy_samples(states, policies, values, weights, max_weight):
    """Replace each sample heavier than max_weight with ceil(w / max_weight) copies that
    share its weight. The objective is unchanged; the copies land in different batches.
    Weights are renormalized to mean 1."""
    weights = np.asarray(weights, dtype=np.float32)
    repeats = np.maximum(np.ceil(weights / max_weight), 1).astype(np.int64)
    if (repeats == 1).all():
        return states, policies, values, weights
    index = np.repeat(np.arange(len(weights)), repeats)
    split_weights = np.repeat(weights / repeats, repeats)
    split_weights *= len(split_weights) / split_weights.sum()
    return ([states[i] for i in index], [policies[i] for i in index], [values[i] for i in index],
            split_weights.astype(np.float32))


class Create_Train_Test_Split:
    def __init__(self,
                 parent_path,
                 num_previous_generations,
                 train_percent=1.0,
                 train_decay=0.95,
                 test_percent=0.1,
                 test_decay=0.95,
                 dedup_average=False,
                 max_sample_weight=None):
        self.parent_path = parent_path
        np.random.seed()

        self.generation = max([int(Path(path).name) for path in glob(self.parent_path + "/*")])

        self.files = sorted([path + "/Self_Play_Data.h5" for path in glob(self.parent_path + "/*")], key=lambda path: int(Path(path).parent.name))

        if self.generation >= 1:  # gen 0 has no network, drop it from every later window
            self.files = [f for f in self.files if int(Path(f).parent.name) != 0]
        self.files = self.files[-num_previous_generations - 1:]
        self.files = self.files[::-1] # reverse it for iteration

        self.train_percent = train_percent
        self.train_decay = train_decay
        self.test_percent = test_percent
        self.test_decay = test_decay
        self.dedup_average = dedup_average
        self.max_sample_weight = max_sample_weight


    def split(self):
        print(f"Collecting data from {len(self.files)} generations!")
        self.train_states, self.train_policies, self.train_values = [], [], []
        self.test_states, self.test_policies, self.test_values = [], [], []

        train_percent = self.train_percent
        test_percent = self.test_percent

        total_games = 0

        for file_path in self.files:
            with h5.File(file_path, mode="r") as file:
                generation = Path(file_path).parent.name  # separator agnostic, os.path.join
                # gives backslashes on windows and split("/") silently returned the whole path

                # Self_Play writes one game as `augmentations` CONSECUTIVE dataset triples,
                # so dataset_id = game_id * augmentations + k (2 per game for connect4,
                # 8 for gomoku). game_stats[2] counts unaugmented games, so we can recover it.
                num_datasets = (len(file.keys()) - 1) // 3
                num_games = int(file["game_stats"][2])

                if num_games <= 0 or num_datasets % num_games != 0:
                    warn(f"gen {generation}: {num_datasets} dataset triples for {num_games} games. "
                         f"Treating every dataset as its own game, the test split may leak.")
                    num_games, augmentations = num_datasets, 1
                else:
                    augmentations = num_datasets // num_games

                # The split has to happen over GAMES, not over pooled positions. Splitting
                # positions puts a board in train and its own mirror/rotation in test, so
                # val_loss just tracks train loss and ModelCheckpoint(save_best_only) can
                # never detect overfitting.
                # one fixed draw per game, seeded by its generation, so a game below the base
                # test_percent is held out in every window and is never trained on
                game_draws = np.random.default_rng([int(generation), 0]).random(num_games)
                train_draws = (game_draws - self.test_percent) / max(1.0 - self.test_percent, 1e-12)

                test_game_ids = set(np.flatnonzero(game_draws < test_percent).tolist())
                train_game_ids = set(np.flatnonzero((game_draws >= self.test_percent)
                                                    & (train_draws < train_percent)).tolist())

                for game_id in tqdm(range(num_games), desc=f"Collecting data from gen: {generation}"):
                    is_test = game_id in test_game_ids
                    if not is_test and game_id not in train_game_ids:
                        continue  # dropped by train_percent / test_percent for this generation

                    first_dataset_id = game_id * augmentations

                    if is_test:
                        states, policies, values = self.test_states, self.test_policies, self.test_values
                    else:
                        states, policies, values = self.train_states, self.train_policies, self.train_values

                    # every augmentation of a game follows that game into the same split
                    for dataset_id in range(first_dataset_id, first_dataset_id + augmentations):
                        states += list(file[f"boards_{dataset_id}"])
                        policies += list(file[f"policies_{dataset_id}"])
                        values += list(file[f"values_{dataset_id}"])

                    total_games += 1

            # no shuffle here on purpose, Dataloader.on_epoch_end reshuffles every epoch
            train_percent -= self.train_decay
            test_percent -= self.test_decay

            train_percent = max(train_percent, 0.0)
            test_percent = max(test_percent, 0.0)

        if self.dedup_average:
            num_train, num_test = len(self.train_states), len(self.test_states)
            self.train_states, self.train_policies, self.train_values, train_counts = dedup_average(
                self.train_states, self.train_policies, self.train_values)
            self.test_states, self.test_policies, self.test_values, test_counts = dedup_average(
                self.test_states, self.test_policies, self.test_values)

            # sample weight = occurrence count, normalized to MEAN 1.0 per split side so
            # the effective learning rate stays where it was tuned. Normalizing per side
            # keeps train and val on the same objective, so ModelCheckpoint(val_loss)
            # still compares like with like
            self.train_weights = train_counts * (len(train_counts) / max(1.0, train_counts.sum()))
            self.test_weights = test_counts * (len(test_counts) / max(1.0, test_counts.sum()))

            removed = 1.0 - len(self.train_states) / max(1, num_train)
            print(f"Dedup: train {num_train} -> {len(self.train_states)} ({removed * 100:.1f}% removed), "
                  f"test {num_test} -> {len(self.test_states)}")
            if len(self.train_weights):
                print(f"Sample weights: mean 1.0, max {self.train_weights.max():.1f} "
                      f"(the most repeated position pulls {self.train_weights.max():.0f}x a unique one)")
        else:
            # no dedup -> every stored sample already carries its own frequency
            self.train_weights = np.ones(len(self.train_states), dtype=np.float32)
            self.test_weights = np.ones(len(self.test_states), dtype=np.float32)

        if self.max_sample_weight is not None and len(self.train_weights):
            num_before = len(self.train_states)
            self.train_states, self.train_policies, self.train_values, self.train_weights = split_heavy_samples(
                self.train_states, self.train_policies, self.train_values, self.train_weights,
                self.max_sample_weight)
            print(f"Split samples heavier than {self.max_sample_weight}: train {num_before} -> "
                  f"{len(self.train_states)}, max weight now {self.train_weights.max():.1f}")

        print("-" * 40)
        # print(f"Total Games: {total_games}")
        print(f"Train stats: Total states {len(self.train_values)}")
        print(f"Test stats: Total states {len(self.test_values)}")
        print("-" * 40)
        return ((self.train_states, self.train_policies, self.train_values, self.train_weights),
                (self.test_states, self.test_policies, self.test_values, self.test_weights))


class Dataloader(tf.keras.utils.PyDataset):
    def __init__(self, states, policies, values, batch_size, weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        np.random.seed()
        self.states = states
        self.policies = policies
        self.values = values
        # per-sample loss weights (occurrence counts after dedup). None -> uniform
        self.weights = (np.asarray(weights, dtype=np.float32) if weights is not None
                        else np.ones(len(states), dtype=np.float32))

        self.batch_size = batch_size
        self.num_samples = len(states)

        self.indexes = np.random.permutation(self.num_samples)

        self.on_epoch_end()

    def on_epoch_end(self):
        np.random.seed()
        self.indexes = np.random.permutation(self.num_samples)

        num_samples = len(self.indexes)
        if num_samples % self.batch_size == 0:
            self.batch_indexes = [*self.indexes.reshape((-1, self.batch_size,))]
        elif num_samples < self.batch_size:
            self.batch_indexes = [*self.indexes.reshape((-1, num_samples,))]
        else:
            remainder = num_samples % self.batch_size
            self.batch_indexes = [*self.indexes[:-remainder].reshape((-1, self.batch_size,)), self.indexes[-remainder:]]

    def __len__(self):
        return len(self.batch_indexes)
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
        weights = self.weights[batched_indexes]
        # (inputs, targets, sample_weights) - keras applies the weights inside the
        # compiled loss wrapper, the custom Loss classes need no change. The weight
        # structure mirrors the target structure: same weights for both heads
        return states, (policies, values), (weights, weights)

def Create_Dataset(folder_path,
                   num_previous_generations,
                   train_batch_size,
                   test_batch_size=None,
                   train_percent=1.0,
                   train_decay=0.75,
                   test_percent=0.1,
                   test_decay=0.75,
                   dedup_average=False,
                   max_sample_weight=None):
    ((train_states, train_policies, train_values, train_weights),
     (test_states, test_policies, test_values, test_weights)) = Create_Train_Test_Split(
        folder_path,
        num_previous_generations,
        train_percent,
        train_decay,
        test_percent,
        test_decay,
        dedup_average=dedup_average,
        max_sample_weight=max_sample_weight).split()

    train_dataloader = Dataloader(train_states, train_policies, train_values, train_batch_size,
                                  weights=train_weights)
    test_dataloader = Dataloader(test_states, test_policies, test_values,
                                 train_batch_size if test_batch_size is None else test_batch_size,
                                 weights=test_weights)
    return train_dataloader, test_dataloader

if __name__ == "__main__":
    import time
    parent_path = "Gomoku/Grok_Zero_Train"
    split = Create_Train_Test_Split(parent_path, 3)
    ((train_state, train_policies, train_values, train_weights),
     (test_states, test_policies, test_values, test_weights)) = split.split()

    # board1 = np.zeros((6, 7), dtype=np.uint64)
    # board2 = np.zeros((6, 7), dtype=np.uint64)
    # board2[0][0] = 1
    # print(board1)
    # print(board2)
    #
    # boards = np.stack([board1 for _ in range(10)], 0)
    # s = time.time()
    # for _ in range(1000):
    #     board_hash(boards)
    # print(time.time() - s)

    # print(h1, h2)
    # print(h1 == h2)

    # train_state = np.array(train_state)

    # data_loader = Dataloader(train_state, train_policies, train_values, 1)
    # for batched_states, (batched_policies, batched_values) in data_loader:
    #     print(batched_states[:, :, :, 1])
    #     print(batched_policies[0].reshape((3, 3)))
    #     raise ValueError
    #     if np.sum(batched_states[:, :, :, 1] * batched_policies[0].reshape((3, 3))) != 0:
    #         print(batched_states[:, :, :, 1])
    #         print(batched_policies[0].reshape((3, 3)))
    #         print(batched_values)
    #         raise ValueError("Sth went wrong")