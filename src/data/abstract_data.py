import pandas as pd
import zipfile
import os
import tensorflow as tf
import numpy as np
import datetime
import random
from .data_extraction import DataExtraction
import json

global feature_constants


class AbstractData(DataExtraction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prepare_random_batches()
        # with open('features.json') as f:
        #     feature_constants = json.load(f)
        #     print(feature_constants)

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

    def next_train_batch(self, store_id=1, forecaster="linear regressor", batch_size=50):
        """
        :param forecastor: Type of forecastor for batch selection
        :param batch_size: Number of rows, ignored if self.is_time_series is True
        :return X: nd.array of shape (batch_size, #features)
        :return y: nd.array of shape (batch_size, 1)
        """
        # if len(self.used_this_epoch) == len(self.train_row_ids):
        #     self._new_epoch()
        # else:
        #     self.is_new_epoch = False
        # batch_size = min(batch_size, len(self.train_row_ids) - len(self.used_this_epoch))
        # row_ids = random.sample(self.train_row_ids - self.used_this_epoch, batch_size)
        # self.used_this_epoch = self.used_this_epoch.union(set(row_ids))
        # return self._extract_rows(row_ids)
        # with open('src/features.json') as f:
        #     feature_constants = json.load(f)
        # print(feature_constants)

        dataExtract = DataExtraction()
        df = dataExtract.train
        # print(df)
        # print(feature_constants["store_type"][0])
        # df = df[df["Store"] == store_id].iloc[:batch_size]

        # print(df)
        print(df[df["Store"] == store_id].iloc[:batch_size])

        # self.X_val, self.y_val = df[df["Store"] == store_id].iloc[:batch_size]
        # print(self.X_val, self.y_val)
        # return self.X_val, self.y_val

        # print(df.loc[ :batch_size ,df["Store"]== 3])
        # print(dataExtract._extract_rows([2]))
        return df[df["Store"] == store_id].iloc[:batch_size]

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

    def validation_batches(self):
        pass
