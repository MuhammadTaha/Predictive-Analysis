try:
    from data_extraction import *
except ModuleNotFoundError:
    from .data_extraction import *
import numpy as np
import os

BATCH_SIZE = 50

# extract all batches and save them

class Data():
    def __init__(self, update_disk=False):
        self.batches_X, self.batches_y = None, None
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

    def extract(self):
        data_extract = DataExtraction()

        # generate all batches
        df = data_extract.train
        store_ids = df.Store.unique()

        self.batches_X, self.batches_y = [], []
        for store_id in [1, 2]: #store_ids:
            row_ids = df.index[df.Store==store_id]
            days, X, y = data_extract.extract_rows_and_days(row_ids)
            days, X, y = days[::-1], X[::-1], y[::-1]
            print("days of this store: ", days)

            start = 0
            while start < days[-1]-BATCH_SIZE:
                end = min(start + BATCH_SIZE, len(days))

                # check if dates are missing and set end to where it's missing
                if np.all(days[start:end] != np.array(list(range(days[start], days[end-1]+1)))[:end-start]):
                    end = start + np.where(days[start:end] != np.array(list(range(days[start], days[end-1]+1)))[:end-start])[0][0] + 1

                self.batches_X.append(X[start:end])
                self.batches_y.append(y[start:end])
                start = end

        self.batches_X, self.batches_y = np.array(self.batches_X), np.array(self.batches_y)
        self.save(os.path.join(DATA_DIR, "extracted"))

    def save(self, path):
        np.save(os.path.join(path, "X"), self.batches_X)
        self.batches_y.save(os.path.join(path, "y"))

    def load(self, path):
        self.batches_X = np.load(os.path.join(path, "X"))
        self.batches_y = np.load(os.path.join(path, "y"))

    def new_epoch(self):
        self.epochs += 1
        self.used_this_epoch = set()

    def train_test_split(self, train_batch_ids, test_batch_ids):
        assert len(train_batch_ids.intersection(test_batch_ids)) == 0
        self.train_batch_ids, self.test_batch_ids = train_batch_ids, test_batch_ids
        self.new_epoch()

    def next_batch(self):
        if self.train_batch_ids - self.used_this_epoch == set():
            self.new_epoch
        batch_id = (self.train_batch_ids - self.used_this_epoch).choice()
        self.used_this_epoch.add(batch_id)
        return self.batches_X[batch_id], self.batches_y[batch_id]
