import pandas as pd
import zipfile
import os
import tensorflow as tf
import numpy as np
import datetime
import random
# from forecaster import AbstractForecaster

import pdb

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
    def __init__(self, data_dir=DATA_DIR, toy=False, keep_zero_sales=False):
        """
        :param data_dir: location of data.zip
        :param toy: if True, take only data of the first 10 stores for model development
        Extracts the data and saves the row_ids for train, val and test data
        Features will be extracted when a certain row is requested in order to save memory
        """

        self.data_dir = data_dir

        # check if files are extracted
        if not set(os.listdir(data_dir)) >= set(["sample_submission.csv", "store.csv", "test.csv", "train.csv"]):
            print("unzip data.zip")
            DataExtraction.extract(data_dir + "/data.zip", data_dir)

        # load into pandas
        self.store = pd.read_csv(data_dir + "/store.csv")
        self.final_test = pd.read_csv(data_dir + "/test.csv")
        self.train = pd.read_csv(data_dir + "/train.csv")

        if toy:
            self.train = self.train.loc[self.train.Store < 3]

        # clean stores with no sales and closed
        #if not keep_zero_sales:
        #    self.train = self.train[(self.train["Open"] != 0) & (self.train['Sales'] != 0)]

        self.time_count = self.train.shape[0]
        self.store_count = self.store.shape[0]
        self.date_keys = sorted(self.train.Date.unique())

        self.features_count = len(self._extract_row(1))

    def _extract_label(self, row_id):
        #  extracts the sales from the specified row
        return [self.train.iloc[row_id]["Sales"]]

    def _extract_rows(self, row_ids):
        X = np.array([self._extract_row(i) for i in row_ids])
        y = np.array([self._extract_label(i) for i in row_ids])

    def extract_rows_and_days(self, row_ids):
        X = np.array([self._extract_row(i) for i in row_ids])
        y = np.array([self._extract_label(i) for i in row_ids])
        start_date = self.str_to_date(self.train.iloc[-1].Date)
        days = np.array([(self.str_to_date(self.train.iloc[row_id].Date) - start_date).days for row_id in row_ids])
        return days, X, y

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
        try:
            row = self.train.iloc[row_id]
        except:
            print(row_id)
            pdb.set_trace()
        return self._extract_loaded_row(row)

    def _extract_loaded_row(self, row):
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

        store_id = row["Store"]
        store = self.store.iloc[store_id - 1]

        curr_date = self.str_to_date(row["Date"])

        store_type = abcd[store["StoreType"]]
        assortment = abc[store["Assortment"]]
        competition_distance = self._competition_distance(store["CompetitionDistance"])
        competition_open = self._competition_open(store, curr_date)
        promo_since_days = self._promo_since_days(store, curr_date)
        days_since_interval = self._promo_interval_since_days(store, curr_date) if promo_since_days > 0 else -1

        day_of_week = np.eye(7)[row["DayOfWeek"] - 1]
        open = row["Open"]
        promo = row["Promo"]
        state_holiday = row["StateHoliday"]
        if state_holiday not in abcd.keys(): state_holiday = "d"
        state_holiday = abcd[state_holiday]
        school_holiday = row["SchoolHoliday"]

        weekday_store_avg = self._weekday_store_avg(row)
        week_of_year_avg = self._week_of_year_avg(row)

        # features = store_type + assortment + [competition_distance, competition_open, promo_since_days,
        #          days_since_interval] + day_of_week + [open, promo] + state_holiday + [school_holiday]
        features = np.concatenate((store_type, assortment, [competition_distance, competition_open, promo_since_days,
                                                            days_since_interval], day_of_week, [open, promo],
                                   state_holiday, [school_holiday, weekday_store_avg, week_of_year_avg]))
        return features

    def str_to_date(self, date_str):
        return datetime.datetime.strptime(date_str, '%Y-%m-%d').date()

    def _weekday_store_avg(self, row):
        if row["Open"] == 0: return 0

        weekday = row["DayOfWeek"]
        store_id = row["Store"]
        avg = np.mean(self.train.loc[(self.train.Store == store_id)
                                       & (self.train.DayOfWeek == weekday)
                                       & (self.train.Open == 1)]["Sales"])

        return avg if avg is not np.isnan(avg) else 0

    def _week_of_year_avg(self, row):
        store_id = row["Store"]

        def get_dates():
            date = self.str_to_date(row["Date"])
            dates = []
            for offset in range(-3, 4):
                mm_dd = (date + datetime.timedelta(days=offset)).isoformat()[5:]  # isoformat is "YYYY-MM-DD"
                for YYYY in ["2013", "2014", "2015"]:
                    dates.append(YYYY + "-" + mm_dd)
            return dates

        dates = get_dates()
        avg = np.mean(self.train.loc[(self.train.Store == store_id)
                      & self.train.Date.isin(dates)
                      & (self.train.Open == 1)]["Sales"])

        return avg if avg is not np.isnan(avg) else 0

    def _competition_distance(self, value):
        # handles missing values for CompetitionDistance
        # if value is not None and not np.isnan(value): return value
        if not self._values_missing(value): return value
        return 100  # TODO

    def _competition_open(self, store, curr_date):
        # calculates the days since competition is open, handles missing values
        month, year = store["CompetitionOpenSinceMonth"], store["CompetitionOpenSinceYear"]

        if self._values_missing(month, year):
            return 100  # TODO

        date = datetime.date(day=1, month=int(month), year=int(year))
        return (curr_date - date).days

    def _promo_since_days(self, store, curr_date):
        if store["Promo2"] == 0: return -1
        week, year = store["Promo2SinceWeek"], store["Promo2SinceYear"]
        if self._values_missing(week, year):
            print("unexpected null values")
            pdb.set_trace()

        date = datetime.date(day=1, month=1, year=int(year)) + datetime.timedelta(weeks=int(week))
        return max((curr_date - date).days, -1)  # if at curr_date the store wasn't participating, return -1

    def _promo_interval_since_days(self, store, curr_date):
        interval_str = store["PromoInterval"]
        starting = interval_str.split(",")
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"]
        starting = [months.index(i) + 1 for i in starting]
        started = [datetime.date(day=1, month=i, year=curr_date.year) for i in starting if i <= curr_date.month] + \
                  [datetime.date(day=1, month=i, year=curr_date.year - 1) for i in starting if i > curr_date.month]
        return min([curr_date - i for i in started]).days

    def _values_missing(self, *args):
        return any([a is None for a in args]) or np.any(np.isnan(args))

    @staticmethod
    def extract(path="data/data.zip", dest="data"):
        print("Extract from {}".format(path))
        zip_ref = zipfile.ZipFile(path, 'r')
        zip_ref.extractall(dest)
        zip_ref.close()
