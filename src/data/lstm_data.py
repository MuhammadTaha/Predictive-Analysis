try:
    from data_extraction import *
except ModuleNotFoundError:
    from .data_extraction import *
import numpy as np
import os
import random

import pdb

BATCH_SIZE = 50

# extract all batches and save them

class LSTMData():
    def __init__(self, timesteps_per_point=10, batch_size=50, update_disk=False, is_debug=False, update_cache=False):
        self.num_tsteps = timesteps_per_point
        self.is_debug = is_debug
        self.batch_size = batch_size
        self.store_ids = None
        self.update_disk = update_disk
        self.update_cache= update_cache
        export_dir = os.path.join(DATA_DIR, "lstm_extracted")
        try:
            assert not update_disk
            self.load(export_dir)
        except (FileNotFoundError, AssertionError) as e:
            self.data_extract = DataExtraction()
            self.days_data, self.sales, self.batch_info = self.extract()
            self.save(export_dir)

        self.epochs = 0
        self.used_this_epoch = set()
        self.train_point_ids = set()
        self.test_point_ids = set()
        self.is_new_epoch = None
        self.features_count = np.array(self.get_point(0)[0]).shape[1]  # of 1st (X,y) pair, take 2nd element of X.shape

    def extract(self):
        print("... Extracting the data")
        # returns: X, Y, batch_info
        # X: store_id -> day_id -> [features] # ordered by day_id
        # Y: store_id -> day_id -> sale # ordered accordingly
        # batch_info: batch_id -> (store_id, day_id)

        X, Y, batch_info = [], [], []

        # generate all batches
        df = self.data_extract.train
        self.store_ids = list(df.Store.unique())
        if self.is_debug:
            self.store_ids = [2, 3, 90]

        for store_id in self.store_ids:
            store_data, store_sales, new_batch_info = self.extract_store(store_id)
            X.append(store_data)
            Y.append(store_sales)
            batch_info += new_batch_info

        return X, Y, batch_info

    def extract_store(self, store_id):
        # returns: store_data, store_sales, batch_info
        # store_data: day_id -> [features] ; ordered by day
        # store_sales: day_id -> sales ; ordered by day
        cache_dir = os.path.join(DATA_DIR, "lstm_store_{}".format(store_id))
        try:
            days, store_data, store_sales = self.load_cached_store(cache_dir)
            print("... Loaded cached data for store", store_id)
        except (AssertionError, FileNotFoundError) as e:
            print("... Extracting store", store_id, e)
            df = self.data_extract.train
            row_ids = df.index[df.Store == store_id]
            days, store_data, store_sales = self.data_extract.extract_rows_and_days(row_ids)
            days, store_data, store_sales = days[::-1], store_data[::-1], store_sales[::-1]
            self.cache_store(cache_dir, days, store_data, store_sales)

        batch_info = [(store_id, day_id) for day_id, _ in enumerate(days)
                        if day_id > self.num_tsteps and np.all( np.array(days[day_id-self.num_tsteps+2:day_id+1] \
                                - np.array(days[day_id-self.num_tsteps+1:day_id]))
                                == np.ones(self.num_tsteps-1))
                    ]

        return store_data, store_sales, batch_info

    def load_cached_store(self, path):
        assert not self.update_cache
        days = np.load(os.path.join(path, "days.npy"))
        store_data = np.load(os.path.join(path, "store_data.npy"))
        store_sales = np.load(os.path.join(path, "store_sales.npy"))
        return days, store_data, store_sales

    def cache_store(self, path, days, store_data, store_sales):
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "days"), days)
        np.save(os.path.join(path, "store_data"), store_data)
        np.save(os.path.join(path, "store_sales"),store_sales)

    def get_point(self, point_id):
        # returns X: (time_steps, #features), Y: scalar
        store_id, day_id = self.batch_info[point_id]
        x = self.days_data[store_id][day_id - self.num_tsteps + 1:day_id+1]
        y = self.sales[store_id][day_id]
        return x, y

    def read_test_csv(self):
        """
            test_data = store_id -> day_id -> [features]
            -> merge train and test data
            run get_batch once for each store/day of the test data
        """
        pass

    def all_train_data(self):
        X, Y = [], []
        for point_id in self.train_point_ids:
            x, y = self.get_point(point_id)
            X.append(x)
            Y.append(y)
        return self.check(X, Y)

    def all_test_data(self):
        X, Y = [], []
        for point_id in self.test_point_ids:
            x, y = self.get_point(point_id)
            X.append(x)
            Y.append(y)
        return self.check(X, Y)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "X"), self.days_data)
        np.save(os.path.join(path, "y"), self.sales)
        np.save(os.path.join(path, "batch_info"), self.batch_info)

    def load(self, path):
        self.days_data = np.load(os.path.join(path, "X.npy"))
        self.sales = np.load(os.path.join(path, "y.npy"))
        self.batch_info = np.load(os.path.join(path, "batch_info.npy"))

    def new_epoch(self):
        self.epochs += 1
        self.used_this_epoch = set()
        self.is_new_epoch = True
        print("START EPOCH", self.epochs)

    def train_test_split(self, train_point_ids, test_point_ids):
        train_point_ids, test_point_ids = set(train_point_ids), set(test_point_ids)
        assert len(train_point_ids.intersection(test_point_ids)) == 0
        self.train_point_ids, self.test_point_ids = train_point_ids, test_point_ids
        self.new_epoch()
        self.X_val, self.y_val = self.all_test_data()

    def check(self, X, Y):
        X, Y = np.array(X), np.array(Y)
        assert np.all(np.isfinite(X)), "X contains bad values: {}".format(np.where(not np.isfinite(X)))
        assert np.all(np.isfinite(Y)), "Y contains bad values: {}".format(np.where(not np.isfinite(Y)))
        return X, Y

    def next_train_batch(self):
        self.is_new_epoch = False
        if len(self.train_point_ids - self.used_this_epoch) < self.batch_size:
            self.new_epoch()

        point_ids = random.sample(list(self.train_point_ids - self.used_this_epoch), k=self.batch_size)
        self.used_this_epoch.update(point_ids)

        X, Y = [], []
        for point_id in point_ids:
            x, y = self.get_point(point_id)
            X.append(x)
            Y.append(y)

        return self.check(X, Y)
