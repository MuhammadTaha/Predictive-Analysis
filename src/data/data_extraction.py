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

OPEN = None

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
        try:
            self.data = pd.read_csv(data_dir + "/processed_data.csv")
            print("We don't load the test data, we need to change the implementation for the final prediction")
        except FileNotFoundError:
            self.prepare_data_for_extraction()
            self.apply_feature_transformation()
            self.apply_feature_transformation_test()
            self.data.to_csv(data_dir + "/processed_data.csv")

        for col_name in self.data.columns:
            if col_name in self.final_test.columns or col_name=="Sales":
                continue
            self.data.drop(col_name, axis=1, inplace=True)

        # check where scalar cols and where list cols (one hot features) are
        # They will be flattened with the scalar columns first
        rows = self.data.iloc[[0]].drop(['index'], axis=1)
        X = rows.drop(['Sales', 'Date'], axis=1).values
        self.scalar_cols = [id for id in range(X.shape[1]) if not isinstance(X[0,id], (np.ndarray, list,)) ]
        self.list_cols = [id for id in range(X.shape[1]) if isinstance(X[0,id], (np.ndarray, list,)) ]

        try:
            assert np.all(self._extract_rows(range(100))[0][:,OPEN] == self.data.Open.values[:100])
        except:
            print("Extracted row OPEN vs data.Open: ",
                  self._extract_rows(range(100))[0][:, OPEN] == self.data.values.Open[:100])
            pdb.set_trace()

        print("OPEN IS ", OPEN, "GO AND TELL THE MODULES\n and change it in abstract_forecaster if its not 0 anymore")

        self.normalize()
        print("Look at this data:")
        print(self.data)
        print(self.data.info())

        self.time_count = self.train.shape[0]
        self.store_count = self.store.shape[0]
        self.date_keys = sorted(self.train.Date.unique())
        self.features_count = self._extract_rows([1])[0].shape[1]

    def prepare_data_for_extraction(self):
        # Dropping features with high missing values percentage
        self.store.drop(['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek',
                         'Promo2SinceYear', 'PromoInterval'], axis=1, inplace=True)
        # replace missing values by median
        self.store.CompetitionDistance.fillna(self._competition_distance_median, inplace=True)

        # don't remove any dates, this makes no sense with the lstm

        # add dates information # Why should they be useful?
        # self.train['Year'] = self.train.Date.dt.year
        # self.train['Month'] = self.train.Date.dt.month
        # self.train['Day'] = self.train.Date.dt.day
        # self.train['WeekOfYear'] = self.train.Date.dt.weekofyear
        self.train.drop('Date', axis=1)
        self.train.reset_index(inplace=True)

        # add dates information test data
        # self.final_test['Year'] = self.final_test.Date.dt.year
        # self.final_test['Month'] = self.final_test.Date.dt.month
        # self.final_test['Day'] = self.final_test.Date.dt.day
        # self.final_test['WeekOfYear'] = self.final_test.Date.dt.weekofyear
        self.final_test.drop('Id', axis=1)
        # self.final_test.drop('Date', axis=1)
        self.final_test.reset_index(inplace=True)

    def _extract_rows(self, row_ids):
        row_ids = list(row_ids)
        rows = self.data.iloc[row_ids].drop(['index'], axis=1)
        X = rows.drop(['Sales', 'Date'], axis=1).values
        Xlists = [X[:, self.scalar_cols]] + \
            [ np.reshape(np.concatenate(X[:, col]), [X.shape[0], -1]) for col in self.list_cols]
        try:
            X = np.concatenate(
               Xlists,
                axis=1
            )
        except Exception as e:
            print(type(e), e)
            pdb.set_trace()
            print(Xlists)
        y = rows.Sales.values
        if not np.all(np.isfinite(y)):
            pdb.set_trace()
            print("y not finite", y)

        return X, y

    def extract_rows_and_days(self, row_ids):
        rows = self.data.iloc[row_ids].drop(['index'], axis=1)
        try:
            X = rows.drop(['Sales', 'Date'], axis=1).values
            Xlists = [X[:, self.scalar_cols]] + \
                     [np.reshape(np.concatenate(X[:, col]), [X.shape[0], -1]) for col in self.list_cols]
            X = np.concatenate(
               Xlists,
                axis=1
            )
            y = rows.Sales.values
            start_date = self.train.iloc[-1].Date
            days = rows.Date.apply(lambda x: (x - start_date).days).values
        except Exception as e:
            print(type(e), e); import pdb
        return days, X, y

    def apply_feature_transformation(self):
        abcd = {
            "a": np.array([1, 0, 0, 0]),
            "b": np.array([0, 1, 0, 0]),
            "c": np.array([0, 0, 1, 0]),
            "d": np.array([0, 0, 0, 1])
        }
        abc = {
            "a": np.array([1, 0, 0]),
            "b": np.array([0, 1, 0]),
            "c": np.array([0, 0, 1])
        }

        sales_avg = self.train[['DayOfWeek', 'Store', 'Sales']]. \
            groupby(['DayOfWeek', 'Store']) \
            .apply(lambda x: 0 * x + np.mean(x))['Sales']
        self.train["WeekdayStoreAvg"] = sales_avg

        self.data = pd.merge(self.train, self.store, how='left', on='Store')
        self.data.StoreType = self.data.StoreType.apply(lambda x: abcd[x])
        self.data.Assortment = self.data.Assortment.apply(lambda x: abc[x])
        self.data.DayOfWeek = self.data.DayOfWeek.apply(lambda x: np.eye(7)[x - 1])
        self.data.StateHoliday = self.data.StateHoliday.apply(lambda x: abcd["d"] if x not in abcd.keys() else abcd[x])

        # Here we decide the order of columns, change carefully
        other_cols = self.data.columns.tolist()
        other_cols.remove("Open")
        self.data = self.data[["Open"]+other_cols]
        global OPEN
        OPEN = 0
        self.open = OPEN

        # self.data.Sales = self.data.Sales.apply(lambda x: np.log(x) + 1)
        #  this gives infinity for the closed days, we need them for the lstm
        # We can add in the model that it predicts logs, I would suggest we
        # just scale it down so that it's in the range [0, 1+a bit]
        # Scaling doesn't effect the percentage loss that we optimize, taking the log does


        #
        # # adding avg sales to data frame
        #
        # sales_avg = sales_avg.rename(columns={'Sales': 'AvgSales'})
        # sales_avg = sales_avg.reset_index()
        # self.data['sales_key'] = self.data['Year'].map(str) + self.data['Month'].map(str) + self.data['Store'].map(str)
        # sales_avg['sales_key'] = sales_avg['Year'].map(str) + sales_avg['Month'].map(str) + sales_avg['Store'].map(str)
        # sales_avg = sales_avg.drop(['Year', 'Month', 'Store'], axis=1)
        # self.data = pd.merge(self.data, sales_avg, how='left', on=('sales_key'))
        # #
        # # adding avg customers to data frame
        # cust = self.data[['Year', 'Month', 'Store', 'Customers']].groupby(['Year', 'Month', 'Store']).mean()
        # cust = cust.rename(columns={'Customers': 'AvgCustomer'})
        # cust = cust.reset_index()
        # self.data['cust_key'] = self.data['Year'].map(str) + self.data['Month'].map(str) + self.data['Store'].map(str)
        # cust['cust_key'] = cust['Year'].map(str) + cust['Month'].map(str) + cust['Store'].map(str)
        # self.data = self.data.drop('Customers', axis=1)  # drop extra columns
        # cust = cust.drop(['Year', 'Month', 'Store'], axis=1)
        # #
        # self.data = pd.merge(self.data, cust, how="left", on=('cust_key'))
        # self.data = self.data.drop(['cust_key', 'sales_key'], axis=1)

    def normalize(self):
        # Find range and save them
        self.x_mean = self.data.mean(axis=0)
        self.x_std = self.data.std(axis=0)
        self.y_mean = self.data.Sales.mean()

        for col_name in self.data.columns:
            if col_name in ["Sales", "Index", "Date", "Store", "Open"]: continue
            if isinstance(self.data.iloc[0][col_name], (list, np.ndarray, )):
                print("This is no scalar:", self.data.iloc[0][col_name], type(self.data.iloc[0][col_name]))
                continue
            try:
                mean = self.data[col_name].mean()
                stddev = self.data[col_name].std()
                self.data[col_name] = (self.data[col_name] - mean) / stddev
                self.final_test[col_name] = (self.final_test[col_name] - mean)/stddev
            except Exception as e:
                print(type(e), e)
                pdb.set_trace()
        self.sales_scaled = self.data.Sales.mean()
        self.data.Sales = self.data.Sales / self.data.Sales.mean()

        # pdb.set_trace()
        # print("calc the std devs and look at the data!")
        # print(self.data.info())



    def apply_feature_transformation_test(self):
        abcd = {
            "a": [1,0,0,0],
            "b": [0,1,0,0],
            "c": [0,0,1,0],
            "d": [0,0,0,1]
        }
        abc = {
            "a": [1,0,0],
            "b": [0,1,0],
            "c": [0,0,1]
        }
        print(
            """
            TODO add the weekday avg to test data
            sales_avg = self.train[['DayOfWeek', 'Store', 'Sales']]. \
                groupby(['DayOfWeek', 'Store']) \
                .apply(lambda x: 0 * x + np.mean(x))['Sales']
            self.train["WeekdayStoreAvg"] = sales_avg
            """
        )

        self.final_test = pd.merge(self.final_test, self.store, how='left', on='Store')
        self.final_test.StoreType = self.final_test.StoreType.apply(lambda x: abcd[x])
        self.final_test.Assortment = self.final_test.Assortment.apply(lambda x: abc[x])
        self.final_test.StateHoliday = self.final_test.StateHoliday.apply(
            lambda x: abcd["d"] if x not in abcd.keys() else abcd[x])
        # self.final_test.Sales = self.final_test.Sales.apply(lambda x: np.log(x) + 1)
        # adding avg sales to data frame
        # sales_avg = self.final_test[['Year', 'Month', 'Store', 'Sales']].groupby(['Year', 'Month', 'Store']).mean()
        # sales_avg = sales_avg.rename(columns={'Sales': 'AvgSales'})
        # sales_avg = sales_avg.reset_index()
        # self.final_test['sales_key'] = self.final_test['Year'].map(str) + self.final_test['Month'].map(str) + \
        #                                self.final_test['Store'].map(str)
        # sales_avg['sales_key'] = sales_avg['Year'].map(str) + sales_avg['Month'].map(str) + sales_avg['Store'].map(str)
        # sales_avg = sales_avg.drop(['Year', 'Month', 'Store'], axis=1)
        # self.final_test = pd.merge(self.final_test, sales_avg, how='left', on=('sales_key'))
        # #
        # # adding avg customers to data frame
        # cust = self.final_test[['Year', 'Month', 'Store', 'Customers']].groupby(['Year', 'Month', 'Store']).mean()
        # cust = cust.rename(columns={'Customers': 'AvgCustomer'})
        # cust = cust.reset_index()
        # self.final_test['cust_key'] = self.final_test['Year'].map(str) + self.final_test['Month'].map(str) + \
        #                               self.final_test['Store'].map(str)
        # cust['cust_key'] = cust['Year'].map(str) + cust['Month'].map(str) + cust['Store'].map(str)
        # self.final_test = self.final_test.drop('Customers', axis=1)  # drop extra columns
        # cust = cust.drop(['Year', 'Month', 'Store'], axis=1)
        # #
        # self.final_test = pd.merge(self.final_test, cust, how="left", on=('cust_key'))
        # self.final_test = self.final_test.drop(['cust_key', 'sales_key'], axis=1)

    def _values_missing(self, *args):
        return any([a is None for a in args]) or np.any(np.isnan(args))

    @staticmethod
    def extract(path="data/data.zip", dest="data"):
        print("Extract from {}".format(path))
        zip_ref = zipfile.ZipFile(path, 'r')
        zip_ref.extractall(dest)
        zip_ref.close()
