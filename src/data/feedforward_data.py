import os
import random

from sklearn.externals import joblib

try:
    from src.data.data_extraction import  DataExtraction, DATA_DIR, DATA_PICKLE_FILE
except ModuleNotFoundError:
    from .data_extraction import DataExtraction
    from .data_extraction import DATA_DIR, DATA_PICKLE_FILE


class FeedForwardData(DataExtraction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prepare_random_batches()

    def _prepare_random_batches(self):
        # splits the rows into train, val and test rows
        train_count = int(self.p_train * self.time_count)
        val_count = int(self.p_val * self.time_count)
        self.train_row_ids = set(range(train_count))
        self.val_row_ids = set(range(train_count, train_count + val_count))
        self.test_row_ids = set(range(train_count + val_count, self.time_count))
        self.used_this_epoch = set()

        # create val data
        self.X_val, self.y_val = self._extract_rows(self.val_row_ids)

        # create test data
        # self.X_test, self.y_test = self._extract_rows(self.test_row_ids)

    def next_train_batch(self, batch_size=50):
        """
        :param batch_size: Number of rows, ignored if self.is_time_series is True
        :return X: nd.array of shape (batch_size, #features)
        :return y: nd.array of shape (batch_size, 1)
        """
        if len(self.used_this_epoch) == len(self.train_row_ids):
            self._new_epoch()
        else:
            self.is_new_epoch = False
        batch_size = min(batch_size, len(self.train_row_ids) - len(self.used_this_epoch))
        row_ids = random.sample(self.train_row_ids - self.used_this_epoch, batch_size)
        self.used_this_epoch = self.used_this_epoch.union(set(row_ids))
        return self._extract_rows(row_ids)

    def get_val_data(self):
        """
        :return X: nd.array of shape (val_set_size, #features)
        :return y: nd.array of shape (batch_size, 1)
        """
        return self.X_val, self.y_val

    def get_test_data(self):
        """
        :return X: nd.array of shape (test_set_size, #features)
        :return y: nd.array of shape (batch_size, 1)
        """
        return self.X_test, self.y_test

    def train_test_split(self, train_point_ids, test_point_ids):
        pass

    def all_train_data(self):
        pass

    def all_test_data(self):
        pass