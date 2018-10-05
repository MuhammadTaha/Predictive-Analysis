try:
    from src.data.data_extraction import *
except ModuleNotFoundError:
    from .data_extraction import *
import numpy as np
import os
import random

import pdb

BATCH_SIZE = 50

# extract all batches and save them

class Data():
    def __init__(self, update_disk=False):
        self.batches_X, self.batches_y, self.store_ids, self.store = None, None, None, None
        # self.store is to look up the store_id of a batch, has shape (#batches)
        try:
            assert not update_disk
            self.load(os.path.join(DATA_DIR, "extracted"))
        except (FileNotFoundError, AssertionError) as e:
            self.extract()

        self.features_count = self.batches_X.shape[2]
        self.epochs = 0
        self.used_this_epoch = set()
        self.train_batch_ids = set()
        self.test_batch_ids = set()
        self.is_new_epoch = None

    def cached_batches_of(self, store_id):
        cached_dir = os.path.join(DATA_DIR, "extracted")
        all_X = np.load(os.path.join(cached_dir, "store_{}_X.npy".format(store_id)))
        all_y = np.load(os.path.join(cached_dir, "store_{}_y.npy".format(store_id)))
        num_samples = all_X.shape[0]
        num_samples -= num_samples % BATCH_SIZE
        num_batches = int(num_samples/BATCH_SIZE)
        all_X = all_X[:num_samples]
        all_y = all_y[:num_samples]
        batches_X = np.reshape(all_X, [num_batches, BATCH_SIZE]+list(all_X.shape[1:]))
        batches_y = np.reshape(all_y, [num_batches, BATCH_SIZE]+list(all_y.shape[1:]))
        print("... loaded cached data for store", store_id)
        return batches_X, batches_y

    def load_batches_of(self, store_id):
        try:
            return self.cached_batches_of(store_id)
        except FileNotFoundError as e:
            print("... generate data for store", store_id)

        # generate all batches
        df = self.data_extract.data
        store_ids = df.Store.unique()

        row_ids = df.index[df.Store == store_id]
        days, X, y = self.data_extract.extract_rows_and_days(row_ids)
        days, X, y = days[::-1], X[::-1], y[::-1]

        start = 0
        batches_X, batches_y = [], []
        while start < len(days) - BATCH_SIZE:
            end = min(start + BATCH_SIZE, len(days))

            # check if dates are missing and set end to where it's missing
            if np.all(np.array(days[1:]) - np.array(days[:-1]) != np.ones(len(days) - 1)):
                end = np.where(np.array(days[1:]) - np.array(days[:-1]) != np.ones(len(days) - 1))[0][0]

            batches_X.append(X[end-BATCH_SIZE:end])
            batches_y.append(y[end-BATCH_SIZE:end])
            start = end

        # save batches
        cached_dir = os.path.join(DATA_DIR, "extracted")
        os.makedirs(cached_dir, exist_ok=True)
        batches_X, batches_y = np.array(batches_X), np.array(batches_y)
        all_X = np.reshape(batches_X, [batches_X.shape[0]*batches_X.shape[1]]+list(batches_X.shape[2:]))
        all_y = np.reshape(batches_y, [batches_y.shape[0]*batches_y.shape[1]]+list(batches_y.shape[2:]))
        np.save(os.path.join(cached_dir, "store_{}_X".format(store_id)), all_X)
        np.save(os.path.join(cached_dir, "store_{}_y".format(store_id)), all_y)

        return batches_X, batches_y

    def extract(self):
        self.data_extract = DataExtraction()
        # generate all batches
        df = self.data_extract.data
        self.store_ids = list(df.Store.unique())
        self.store = []

        # for each batch, the store can be checked in self.store (of len: #batches)
        for store_id in self.store_ids:
            batches_X, batches_y = self.load_batches_of(store_id)
            self.store += [store_id]*len(batches_X)
            if self.batches_X is not None:
                self.batches_X = np.concatenate([self.batches_X, batches_X], 0)
                self.batches_y = np.concatenate([self.batches_y, batches_y], 0)
            else:
                self.batches_X, self.batches_y = batches_X, batches_y
        self.store = np.array(self.store)
        self.save(os.path.join(DATA_DIR, "extracted"))

    def all_train_data(self):
        batches = list(self.train_batch_ids)
        return np.concatenate(self.batches_X[batches], axis=0), np.concatenate(self.batches_y[batches], axis=0)

    def all_test_data(self):
        batches = list(self.test_batch_ids)
        return np.concatenate(self.batches_X[batches], axis=0), np.concatenate(self.batches_y[batches], axis=0)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "X"), self.batches_X)
        np.save(os.path.join(path, "y"), self.batches_y)
        np.save(os.path.join(path, "store_ids"), self.store_ids)

    def load(self, path):
        self.batches_X = np.load(os.path.join(path, "X.npy"))
        self.batches_y = np.load(os.path.join(path, "y.npy"))
        self.store_ids = np.load(os.path.join(path, "store_ids.npy"))

    def new_epoch(self):
        self.epochs += 1
        self.used_this_epoch = set()
        self.is_new_epoch = True
        print("START EPOCH", self.epochs)

    def train_test_split(self, train_batch_ids, test_batch_ids):
        assert len(train_batch_ids.intersection(test_batch_ids)) == 0
        self.train_batch_ids, self.test_batch_ids = train_batch_ids, test_batch_ids
        self.new_epoch()
        self.X_val, self.y_val = self.all_test_data()

    def next_train_batch(self):
        self.is_new_epoch = False
        if self.train_batch_ids - self.used_this_epoch == set():
            self.new_epoch()
        batch_id = random.choice(list(self.train_batch_ids - self.used_this_epoch))
        self.used_this_epoch.add(batch_id)
        return self.batches_X[batch_id], self.batches_y[batch_id]
