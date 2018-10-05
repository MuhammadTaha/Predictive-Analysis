import pdb
import zipfile

import numpy as np
import os
import pandas as pd

from src.data.feature_enum import ONE_HOT_FEATURES, DROP_FEATURES

"""
- unzips /data/data.zip if not done yet
- Reads csv data to pandas
- Creates `tf.placeholders` for data batches
- Splits data into train and test set
- Provides a method to fetch the next train data batch
- Estimates missing data
"""

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")
DATA_PICKLE_FILE = 'EXTRACTED_FEATURES'


class DataExtraction:
    def __init__(self, data_dir=DATA_DIR, train_path=None, test_path=None):
        """
        Extracts the data and saves the row_ids for train, val and test data
        Features will be extracted when a certain row is requested in order to save memory
        """

        self.data_dir = data_dir

        # check if files are extracted
        if train_path and test_path:
            self.final_test = pd.read_csv(test_path)
            self.data = pd.read_csv(train_path)
        else:
            self.store = pd.read_csv(data_dir + "/store.csv")
            self.final_test = pd.read_csv(data_dir + "/test.csv", parse_dates=['Date'], )
            self.train = pd.read_csv(data_dir + "/train.csv", parse_dates=['Date'], )
            self._competition_distance_median = self.store['CompetitionDistance'].median()
            self.prepare_data_for_extraction()
            self.apply_feature_transformation()
            self.apply_feature_transformation_test()

    def prepare_data_for_extraction(self):
        # replace missing values by median
        self.store.CompetitionDistance.fillna(self._competition_distance_median, inplace=True)
        self.final_test.fillna(1, inplace=True)

        # remove stores that's not open
        self.train = self.train[self.train['Open'] != 0]
        # self.train = self.train.drop('Open', axis=1)

        # remove stores that't not open test data
        # self.final_test = self.final_test[self.final_test['Open'] != 0]
        # self.final_test = self.final_test.drop('Open', axis=1)

        # remove entries with zero sales
        self.train = self.train[self.train['Sales'] != 0]

        # add dates information
        self.train['Year'] = self.train.Date.dt.year
        self.train['Month'] = self.train.Date.dt.month
        self.train['Day'] = self.train.Date.dt.day
        self.train['WeekOfYear'] = self.train.Date.dt.weekofyear
        # self.train.drop('Date', axis=1)
        self.train.reset_index(inplace=True)

        # add dates information test data
        self.final_test['Year'] = self.final_test.Date.dt.year
        self.final_test['Month'] = self.final_test.Date.dt.month
        self.final_test['Day'] = self.final_test.Date.dt.day
        self.final_test['WeekOfYear'] = self.final_test.Date.dt.weekofyear
        self.final_test.reset_index(inplace=True)

    def apply_feature_transformation(self):
        abcd = {
            "a": [1, 0, 0, 0],
            "b": [0, 1, 0, 0],
            "c": [0, 0, 1, 0],
            "d": [0, 0, 0, 1]
        }
        abc = {
            "a": [1, 0, 0],
            "b": [0, 1, 0],
            "c": [0, 0, 1]
        }
        self.data = pd.merge(self.train, self.store, how='left', on='Store')
        self.data.fillna(0.0, inplace=True)
        self.data.StoreType = self.data.StoreType.apply(lambda x: abcd[x])
        self.data.Assortment = self.data.Assortment.apply(lambda x: abc[x])
        self.data.StateHoliday = self.data.StateHoliday.apply(lambda x: abcd["d"] if x not in abcd.keys() else abcd[x])
        self.data.Sales = self.data.Sales.apply(lambda x: np.log(x) + 1)


        # adding sales avg
        sales_avg = self.data[['DayOfWeek', 'Store', 'Sales']].groupby(['DayOfWeek', 'Store']).mean()
        sales_avg = sales_avg.reset_index()
        self.sales_avg = sales_avg.rename(columns={'Sales': 'AvgSales'})
        self.data = pd.merge(self.data, self.sales_avg, how='left', on=('Store', 'DayOfWeek'))

        # self.data.Sales = self.data.Sales - self.data.AvgSales


        # Transform Hot Encoding Features
        self.data.DayOfWeek = self.data.DayOfWeek.apply(lambda x: np.eye(7)[x - 1])
        for feature, feature_names in ONE_HOT_FEATURES.items():
            self.data[feature_names] = pd.DataFrame(self.data[feature].values.tolist())
        # self.data = self.data.drop(DROP_FEATURES, axis=1)

        self.data = self.data.sample(frac=1).reset_index(drop=True)

    def apply_feature_transformation_test(self):
        abcd = {
            "a": [1, 0, 0, 0],
            "b": [0, 1, 0, 0],
            "c": [0, 0, 1, 0],
            "d": [0, 0, 0, 1]
        }
        abc = {
            "a": [1, 0, 0],
            "b": [0, 1, 0],
            "c": [0, 0, 1]
        }
        self.final_test.fillna(0.0, inplace=True)
        self.final_test = pd.merge(self.final_test, self.store, how='left', on='Store')
        self.final_test.StoreType = self.final_test.StoreType.apply(lambda x: abcd[x])
        self.final_test.Assortment = self.final_test.Assortment.apply(lambda x: abc[x])
        self.final_test.StateHoliday = self.final_test.StateHoliday.apply(
            lambda x: abcd["d"] if x not in abcd.keys() else abcd[x])

        self.final_test = pd.merge(self.final_test, self.sales_avg, how='left', on=('Store', 'DayOfWeek'))

        self.final_test.DayOfWeek = self.final_test.DayOfWeek.apply(lambda x: np.eye(7)[x - 1])
        # Transform Hot Encoding Features
        for feature, feature_names in ONE_HOT_FEATURES.items():
            self.final_test[feature_names] = pd.DataFrame(self.final_test[feature].values.tolist())
        self.final_test = self.final_test.drop(DROP_FEATURES, axis=1)

        self.final_test.fillna(0.0, inplace=True)
        self.final_test = self.final_test.sample(frac=1).reset_index(drop=True)
