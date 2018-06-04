import pandas as pd
import zipfile
import os
import tensorflow as tf
import numpy as np
from datetime import datetime

import pdb

"""
- unzips /data/data.zip if not done yet
- Reads csv data to pandas
- Creates `tf.placeholders` for data batches
- Splits data into train and test set
- Provides a method to fetch the next train data batch
- Estimates missing data
"""

class Data():
    def __init__(self, dir="data", p_train=0.6, p_val=0.2, p_test=0.2, is_time_series=False):
        """
        :param dir: location of data.zip
        :param p_train: percentage of the labeled data used for training
        :param p_val: percentage of the labeled data used for validation
        :param p_test: percentage of the labeled data used for testing
        :param is_time_series: if True, next_train_batch will return time series data for random stores,
            otherwise next_train_batch will return a random batch (mixed stores, not ordered by date)

        Extracts the data and saves the row_ids for train, val and test data
        Features will be extracted when a certain row is requested in order to save memory
        """
        assert p_train+p_val+p_test == 1
        
        self.data_dir = dir
        self.is_time_series = is_time_series

        # check if files are extracted
        if set(os.listdir(dir)) >= set(["sample_submission.csv", "store.csv", "test.csv", "train.csv"]):
            print("Data is extracted already")
        else:
            Data.extract(dir+"/data.zip", dir)

        # load into pandas
        self.store = pd.read_csv(dir+"/store.csv")
        self.final_test = pd.read_csv(dir+"/test.csv")
        self.train = pd.read_csv(dir+"/train.csv")

        self.time_count = self.train.shape[0]
        self.store_count = self.store.shape[0]

        self.test_stuff()

        pdb.set_trace()

        if self.is_time_series:
            return self._prepare_time_series()
        else:
            return self._prepare_random_batches()

    def test_stuff(self):
        # are the dates in order?
        date = [datetime.strptime(d, '%Y-%m-%d') for d in self.train["Date"].tolist()]
        ordered = True
        for i in range(len(date)-1):
            if (date[i] >= date[i+1]) != (date[0] >= date[1]):
                ordered = False
                break
        print("Dates are ordered:", ordered)


    def _prepare_time_series(self):
        # splits the stores into train, val and test stores
        pass

    def _prepare_random_batches(self):
        # splits the rows into train, val and test rows
        pass

    def next_train_batch(self, batch_size=50):
        """
        :param batch_size: Number of rows, ignored if self.is_time_series is True
        :return X: nd.array of shape (batch_size, #features)
        :return y: nd.array of shape (batch_size, 1)
        """
        if self.is_time_series:
            return self._next_time_series()
        else:
            return self._next_random_batch()

    def _get_time_series(self, store_id):
        """
        :param store_id: store for which the time series will be generated
        :return X: nd.array of shape (batch_size, #features)
        :return y: nd.array of shape (batch_size, 1)
        The data returned is only for the specified store, and ordered by date
        """
        pass

    def _next_time_series(self):
        """
        :return X: nd.array of shape (batch_size, #features)
        :return y: nd.array of shape (batch_size, 1)
        Chooses the store and let's _get_time_series do the rest
        """
        pass

    def _next_random_batch(self, batch_size):
        """
        :param batch_size: number of rows
        :return X: nd.array of shape (batch_size, #features)
        :return y: nd.array of shape (batch_size, 1)
        """
        pass

    def _extract_row(self, row):
        """
        :param row: row of self.train to extract features for
        :return: nd.array of shape (#features)
        """
        # look up the store id
        # look up the current date (curr_date)
        # get row of self.stores
        # compute dates of the store features
        # call a handler for missing values (should be easily exchangable)
        # for each date feature: compute the difference to curr_date (in days)
        pass



    @staticmethod
    def extract(path="data/data.zip", dest="data"):
        print("Extract from {}".format(path))
        zip_ref = zipfile.ZipFile(path, 'r')
        zip_ref.extractall(dest)
        zip_ref.close()