import pandas as pd
import zipfile
import os
import tensorflow as tf
import numpy as np
import datetime
import random

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
        self.p_train, self.p_val, self.p_test = p_train, p_val, p_test
        
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

        self.epochs = 0
        self.is_new_epoch = True

        if self.is_time_series:
            self._prepare_time_series()
        else:
            self._prepare_random_batches()

        self.features_count = len(self._extract_row(1))

    def test_stuff(self):
        # test if promo2sincewhatever is missing exactly where promo2==constant

        r = {True: [], False: []} # True: where (promo2 is zero)==(value missing)
        for store_id in range(self.store_count):
            store = self.store.iloc[store_id]
            missing = store["Promo2SinceYear"] is None or store["Promo2SinceWeek"] is None

            r[missing == (store["Promo2"]==0)].append(store_id)

        print("list lemghts", len(r[True]), len(r[False]))
        pdb.set_trace()


        return
        # are the dates in order?
        dates = [datetime.strptime(d, '%Y-%m-%d') for d in self.train["Date"].tolist()]
        ordered = True
        for i in range(len(dates)-1):
            if (dates[i] >= dates[i+1]) != (dates[0] >= dates[1]):
                ordered = False
                break
        print("Dates are ordered:", ordered)


    def _prepare_time_series(self):
        # splits the stores into train, val and test stores
        train_count = int(self.p_train*self.store_count)
        val_count = int(self.p_val*self.store_count)
        self.train_store_ids = set(range(train_count))
        self.val_store_ids = set(range(train_count, train_count+val_count))
        self.test_store_ids = set(range(train_count+val_count, self.store_count))
        self.used_this_epoch = set()

        # create val data
        pass

        # create test data
        pass

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
        #self.X_test, self.y_test = self._extract_rows(self.test_row_ids)

    def _new_epoch(self):
        self.used_this_epoch = set()
        self.epochs += 1
        self.is_new_epoch = True

    def next_train_batch(self, batch_size=50):
        """
        :param batch_size: Number of rows, ignored if self.is_time_series is True
        :return X: nd.array of shape (batch_size, #features)
        :return y: nd.array of shape (batch_size, 1)
        """
        if self.is_time_series:
            return self._next_time_series()
        else:
            return self._next_random_batch(batch_size)

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
        if len(self.used_this_epoch) == len(self.train_store_ids):
            self._new_epoch()
        else:
            self.is_new_epoch = False
        store_id = random.sample(self.train_store_ids-self.used_this_epoch, 1)[0]
        self.used_this_epoch = self.used_this_epoch.union(set([store_id]))
        return self._get_time_series(store_id)

    def _next_random_batch(self, batch_size):
        """
        :param batch_size: number of rows
        :return X: nd.array of shape (batch_size, #features)
        :return y: nd.array of shape (batch_size, 1)
        """
        if len(self.used_this_epoch) == len(self.train_row_ids):
            self._new_epoch()
        else:
            self.is_new_epoch = False
        batch_size = min(batch_size, len(self.train_row_ids) - len(self.used_this_epoch))
        row_ids = random.sample(self.train_row_ids-self.used_this_epoch, batch_size)
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

    def _extract_label(self, row_id):
        # extracts the sales from the specified row
        return [self.train.iloc[row_id]["Sales"]]

    def _extract_rows(self, row_ids):
        X = np.array([self._extract_row(i) for i in row_ids])
        y = np.array([self._extract_label(i) for i in row_ids])
        return X, y

    def _extract_row(self, row_id):
        """
        :param row: row of self.train to extract features for
        :return: nd.array of shape (#features)
        """

        """
        Store
            RangeIndex: 1115 entries, 0 to 1114
            Data columns (total 10 columns):
            Store                        1115 non-null int64
            StoreType                    1115 non-null object
            Assortment                   1115 non-null object
            CompetitionDistance          1112 non-null float64
            CompetitionOpenSinceMonth    761 non-null float64
            CompetitionOpenSinceYear     761 non-null float64
            Promo2                       1115 non-null int64
            Promo2SinceWeek              571 non-null float64
            Promo2SinceYear              571 non-null float64
            PromoInterval                571 non-null object
            dtypes: float64(5), int64(2), object(3)
            
        Train
            RangeIndex: 1017209 entries, 0 to 1017208
            Data columns (total 9 columns):
            Store            1017209 non-null int64
            DayOfWeek        1017209 non-null int64
            Date             1017209 non-null object
            Sales            1017209 non-null int64
            Customers        1017209 non-null int64
            Open             1017209 non-null int64
            Promo            1017209 non-null int64
            StateHoliday     1017209 non-null object
            SchoolHoliday    1017209 non-null int64
            dtypes: int64(7), object(2)
            
        We extract
            Store Type	                One hot 4
            Assortment	                One hot 3
            CompetitionDistance	        float
            CompetitionOpenSinceDays	uint
            PromoSinceDays	            uint if participating in promo, else -1
            DaysSinceInterval	        uint if participating in promo, else -1
            
            DayOfWeek                   One hot 7
            Open                        {0,1}
            Promo                       {0,1}
            StateHoliday                {0, 'b', 'a', '0', 'c'} => one hot 4
            SchoolHoliday	            {0,1}
        """
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


        row = self.train.iloc[row_id]
        store_id = row["Store"]
        store = self.store.iloc[store_id-1]

        curr_date = self.str_to_date(row["Date"])

        store_type = abcd[store["StoreType"]]
        assortment = abc[store["Assortment"]]
        competition_distance = self._competition_distance(store["CompetitionDistance"])
        competition_open = self._competition_open(store, curr_date)
        promo_since_days = self._promo_since_days(store, curr_date)
        days_since_interval = self._promo_interval_since_days(store, curr_date) if promo_since_days > 0 else -1

        day_of_week = np.eye(7)[row["DayOfWeek"]-1]
        open = row["Open"]
        promo = row["Promo"]
        state_holiday = row["StateHoliday"]
        if state_holiday not in abcd.keys(): state_holiday = "d"
        state_holiday = abcd[state_holiday]
        school_holiday = row["SchoolHoliday"]

        #features = store_type + assortment + [competition_distance, competition_open, promo_since_days,
        #          days_since_interval] + day_of_week + [open, promo] + state_holiday + [school_holiday]
        features = np.concatenate((store_type, assortment, [competition_distance, competition_open, promo_since_days,
                  days_since_interval], day_of_week, [open, promo], state_holiday, [school_holiday]))
        return features

    def str_to_date(self, date_str):
        return datetime.datetime.strptime(date_str, '%Y-%m-%d').date()

    def _competition_distance(self, value):
        # handles missing values for CompetitionDistance
        #if value is not None and not np.isnan(value): return value
        if not self._values_missing(value): return value
        return 100 # TODO

    def _competition_open(self, store, curr_date):
        # calculates the days since competition is open, handles missing values
        month, year = store["CompetitionOpenSinceMonth"], store["CompetitionOpenSinceYear"]

        if self._values_missing(month, year):
            return 100 # TODO

        date = datetime.date(day=1, month=int(month), year=int(year))
        return (curr_date - date).days

    def _promo_since_days(self, store, curr_date):
        if store["Promo2"] == 0: return -1
        week, year = store["Promo2SinceWeek"], store["Promo2SinceYear"]
        if self._values_missing(week, year):
            print("unexpected null values")
            pdb.set_trace()

        date = datetime.date(day=1, month=1, year=int(year)) + datetime.timedelta(weeks=int(week))
        return max((curr_date - date).days, -1) # if at curr_date the store wasn't participating, return -1

    def _promo_interval_since_days(self, store, curr_date):
        interval_str = store["PromoInterval"]
        starting = interval_str.split(",")
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"]
        starting = [months.index(i)+1 for i in starting]
        started = [datetime.date(day=1, month=i, year=curr_date.year) for i in starting if i<=curr_date.month] + \
                  [datetime.date(day=1, month=i, year=curr_date.year-1) for i in starting if i>curr_date.month]
        return min([curr_date-i for i in started]).days

    def _values_missing(self, *args):
        return any([a is None for a in args]) or np.any(np.isnan(args))


    @staticmethod
    def extract(path="data/data.zip", dest="data"):
        print("Extract from {}".format(path))
        zip_ref = zipfile.ZipFile(path, 'r')
        zip_ref.extractall(dest)
        zip_ref.close()