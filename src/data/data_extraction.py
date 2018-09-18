import pdb
import zipfile

import numpy as np
import os
import pandas as pd

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
        self.final_test = pd.read_csv(data_dir + "/test.csv", parse_dates=['Date'], )
        self.train = pd.read_csv(data_dir + "/train.csv", parse_dates=['Date'], )

        self._competition_distance_median = self.store['CompetitionDistance'].median()

        if toy:
            self.train = self.train.loc[self.train.Store < 3]

        # clean stores with no sales and closed

        # if not keep_zero_sales:
        #    self.train = self.train[(self.train["Open"] != 0) & (self.train['Sales'] != 0)]


        self.prepare_data_for_extraction()
        self.apply_feature_transformation()

        self.time_count = self.train.shape[0]
        self.store_count = self.store.shape[0]
        self.date_keys = sorted(self.train.Date.unique())
        self.features_count = len(self._extract_row(1))

    def prepare_data_for_extraction(self):
        # Dropping features with high missing values percentage
        self.store.drop(['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek',
                         'Promo2SinceYear', 'PromoInterval'], axis=1, inplace=True)
        # replace missing values by median
        self.store.CompetitionDistance.fillna(self._competition_distance_median, inplace=True)

        # remove stores that's not open
        self.train = self.train[self.train['Open'] != 0]
        self.train = self.train.drop('Open', axis=1)

        # remove entries with zero sales
        self.train = self.train[self.train['Sales'] != 0]

        # add dates information
        self.train['Year'] = self.train.Date.dt.year
        self.train['Month'] = self.train.Date.dt.month
        self.train['Day'] = self.train.Date.dt.day
        self.train['WeekOfYear'] = self.train.Date.dt.weekofyear
        self.train.reset_index(inplace=True)

    def _extract_label(self, row_id):
        #  extracts the sales from the specified row
        return [self.train.iloc[row_id]["Sales"]]

    def _extract_rows(self, row_ids):
        rows = self.data.iloc[row_ids].drop(['index'], axis=1)
        X = rows.drop(['Sales', 'Date'], axis=1).values
        y = rows.Sales.values
        return X, y
    def extract_rows_and_days(self, row_ids):
        rows = self.data.iloc[row_ids].drop(['index'], axis=1)
        X = rows.drop(['Sales', 'Date'], axis=1).values
        y = rows.Sales.values
        start_date = self.data.iloc[-1].Date
        days = rows.Date.apply(lambda x: (x - start_date).days).values
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
            Promo2                      {0,1}
            Store                       int
            
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

    def apply_feature_transformation(self):
        #TODO put the one hot mapping again
        abcd = {
            "a": 0,
            "b": 1,
            "c": 2,
            "d": 3
        }
        abc = {
            "a": 0,
            "b": 1,
            "c": 2
        }

        self.data = pd.merge(self.train, self.store, how='left', on='Store')
        self.data.StoreType = self.data.StoreType.apply(lambda x: abcd[x])
        self.data.Assortment = self.data.Assortment.apply(lambda x: abc[x])
        # self.data.DayOfWeek = self.data.DayOfWeek.apply(lambda x: np.eye(7)[x - 1])
        self.data.StateHoliday = self.data.StateHoliday.apply(lambda x: abcd["d"] if x not in abcd.keys() else abcd[x])
        self.data.Sales = self.data.Sales.apply(lambda x: np.log(x))
        #
        # adding avg sales to data frame
        sales_avg = self.data[['Year', 'Month', 'Store', 'Sales']].groupby(['Year', 'Month', 'Store']).mean()
        sales_avg = sales_avg.rename(columns={'Sales': 'AvgSales'})
        sales_avg = sales_avg.reset_index()
        self.data['sales_key'] = self.data['Year'].map(str) + self.data['Month'].map(str) + self.data['Store'].map(str)
        sales_avg['sales_key'] = sales_avg['Year'].map(str) + sales_avg['Month'].map(str) + sales_avg['Store'].map(str)
        sales_avg = sales_avg.drop(['Year', 'Month', 'Store'], axis=1)
        self.data = pd.merge(self.data, sales_avg, how='left', on=('sales_key'))
        #
        # adding avg customers to data frame
        cust = self.data[['Year', 'Month', 'Store', 'Customers']].groupby(['Year', 'Month', 'Store']).mean()
        cust = cust.rename(columns={'Customers': 'AvgCustomer'})
        cust = cust.reset_index()
        self.data['cust_key'] = self.data['Year'].map(str) + self.data['Month'].map(str) + self.data['Store'].map(str)
        cust['cust_key'] = cust['Year'].map(str) + cust['Month'].map(str) + cust['Store'].map(str)
        self.data = self.data.drop('Customers', axis=1)  # drop extra columns
        cust = cust.drop(['Year', 'Month', 'Store'], axis=1)
        #
        self.data = pd.merge(self.data, cust, how="left", on=('cust_key'))
        self.data = self.data.drop(['cust_key', 'sales_key'], axis=1)

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

        # store features
        store_type = abcd[store["StoreType"]]
        assortment = abc[store["Assortment"]]
        competition_distance = store["CompetitionDistance"]

        day_of_week = np.eye(7)[row["DayOfWeek"] - 1]
        promo = row["Promo"]
        state_holiday = row["StateHoliday"]
        if state_holiday not in abcd.keys(): state_holiday = "d"
        state_holiday = abcd[state_holiday]
        school_holiday = row["SchoolHoliday"]

        year, month, day, WeekOfYear = row.Year, row.Month, row.Day, row.WeekOfYear

        weekday_store_avg = self._weekday_store_avg(row)
        week_of_year_avg = self._week_of_year_avg(row)
        month_store_avg = self._month_store_avg(row)
        promo2 = store['Promo2']
        features = np.concatenate(
            (store_type, assortment, [competition_distance, promo2], day_of_week, [promo], state_holiday,
             [school_holiday, np.log(weekday_store_avg), np.log(week_of_year_avg), year, month, day, WeekOfYear,
              np.log(month_store_avg)]))
        return features

    def _month_store_avg(self, row):
        avg = self.train.Sales[
            (self.train.Store == row.Store) & (self.train.Year == row.Year) & (self.train.Month == row.Month)].mean()
        return np.log(avg) if avg is not np.isnan(avg) else 0

    def _weekday_store_avg(self, row):
        avg = self.train.Sales[(self.train.Store == row.Store) & (self.train.Year == row.Year) & (
            self.train.DayOfWeek == row.DayOfWeek)].mean()
        return np.log(avg) if avg is not np.isnan(avg) else 0

    def _week_of_year_avg(self, row):
        avg = self.train.Sales[(self.train.Store == row.Store) & (self.train.Year == row.Year) & (
            self.train.WeekOfYear == row.WeekOfYear)].mean()
        return np.log(avg) if avg is not np.isnan(avg) else 0

    def _values_missing(self, *args):
        return any([a is None for a in args]) or np.any(np.isnan(args))

    @staticmethod
    def extract(path="data/data.zip", dest="data"):
        print("Extract from {}".format(path))
        zip_ref = zipfile.ZipFile(path, 'r')
        zip_ref.extractall(dest)
        zip_ref.close()
