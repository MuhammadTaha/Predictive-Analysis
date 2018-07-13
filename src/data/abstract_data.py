from .data_extraction import *
import pandas as pd
import zipfile
import os
import tensorflow as tf
import numpy as np
import datetime
import random


class AbstractData(DataExtraction):

    dataExtract = DataExtraction()
    df = dataExtract.train
    df['is_used'] = 0

    # def __init__(self):
    #     # global

    def next_train_batch(self, store_id = None, forecaster= "linear regressor", batch_size=50,start_date = "2015-07-01",end_date="2015-08-01"):
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


        df = self.df
        if store_id is None:
            stores = df["Store"].unique()
            store_id = random.choice(stores)

        result_date = self.random_dates(self,start_date,end_date,1)

        mask = (df['Date'] >= result_date) & (df['Date'] <= end_date)
        df = df.loc[mask]
        result  =  df[(((df["Store"]) == store_id) & (df["Sales"] != 0) & (df["Customers"] != 0) & (df["is_used"] == 0))].iloc[:batch_size]
        result['is_used'] = 1
        self.df.update(result)

        # print(result)
        x = (batch_size,result.values)
        y= (batch_size,store_id)

        return x,y


    def validation_batches(self,forecaster = "linear regressor"):
        # result = df[df["Store"] == store_id].iloc[:batch_size]
        # x = (batch_size, result)
        d = DataExtraction()
        dfinal = d.final_test
        return dfinal.values

    def days_between(self,d1, d2):
        d1 = datetime.datetime.strptime(d1, "%Y-%m-%d")
        d2 = datetime.datetime.strptime(d2, "%Y-%m-%d")
        return abs((d2 - d1).days)

    def random_dates(self,start_date, end_date, n, unit='D', seed=None):
        #n is the number of dates you want in output

        ndays = self.days_between(self,start_date, end_date)
        arr_ndays = list(range(ndays))
        diff = random.choice(arr_ndays)
        # print(diff)
        res_result = pd.to_datetime(start_date) + pd.DateOffset(days=np.int(diff))
        orig_date = str(res_result)

        d = datetime.datetime.strptime(orig_date, '%Y-%m-%d %H:%M:%S')
        d = d.strftime('%Y-%m-%d')
        return d

        # return
        # return datetime.date.start_date + datetime.date(days = np.float(diff))


