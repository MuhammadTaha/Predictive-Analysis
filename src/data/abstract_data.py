from .data_extraction import *
import pandas as pd
import zipfile
import os
import tensorflow as tf
import numpy as np
import datetime
import random


class AbstractData():


    dataExtract = DataExtraction()
    df = dataExtract.train
    df['is_used'] = 0

    train_data = list()

    # def __init__(self,epoch = 1,store_id = None):
    #
    #     stores_data = list()
    #     epoch_data = list()
    #
    #     if store_id is None:
    #         store_id = self.df["Store"].unique()
    #
    #     for x in range(epoch):
    #         for item in store_id:
    #             stores_data = AbstractData.next_train_batch(self,store_id = item,forecaster = "linear regressor" , batch_size= 50)
    #             print(len(stores_data))
    #         epoch_data.append(stores_data)
    #         self.df['is_used'] = 0
    #
    #
    #     print(epoch_data)
    #
    #     print(len(epoch_data))


    # init(store_ids):
    # set the possible store ids so that we can train with subsets of the data (otherwise it takes too long)
    # generate all training batches and save them in a list, the list keys are the batch_ids
    # test_train_split(train_batch_ids, test_batch_ids) set self.train_batch_ids, self.test_batch_ids
    # The actual splits will be done from somewhere else, from the model selection process

    # next_train_batch:
    # select one number of self.train_batch_ids - self.train_batches_used_this_epoch (assuming both are sets, then - does SetA \ SetB )
    # save the chosen batch_id to the set of batch_ids that have been used in the current epoch
    # check if all batches have been used in the current epoch, and start a new epoch if that's the case
    # New epoch means: increase self.epochs by one and set self.train_batches_used_this_epoch = set()


    # each batch shall have data of one single store
    # a list of different batch of data of different store
    # each batch shall contain just the row ids of the data






    # def __init__(self):


    def get_training_data(self,epoch = 1,store_id = None):
        stores_data = list()
        epoch_data = list()

        if store_id is None:
            store_id = self.df["Store"].unique()

        for x in range(epoch):
            for item in store_id:
                stores_data = AbstractData.next_train_batch(self, store_id=item, forecaster="linear regressor",
                                                            batch_size=50)
                print(len(stores_data))
            epoch_data.append(stores_data)
            self.df['is_used'] = 0

        print(epoch_data)

        print(len(epoch_data))

        return epoch_data


    def next_train_batch(self, store_id = None, forecaster= "linear regressor", batch_size=50,start_date = "2013-01-01",end_date="2015-08-01"):#end_date="2015-08-01"):
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

        batches = list()

        df = self.df
        if store_id is None:
            stores = df["Store"].unique()
            store_id = random.choice(stores)


        result_date = self.random_dates(start_date,end_date,1)
        # print(result_date)
        # mask = (df['Date'] >= "2013-01-01") & (df['Date'] <= end_date)
        mask = (df['Date'] >= result_date) & (df['Date'] <= end_date)

        df = df.loc[mask]
        isFinished = False
        while(isFinished is False):
        # for x in range(2):
            if np.isscalar(store_id) is False:
                result  =  df[(((df["Store"]).isin(store_id)) & (df["Sales"] != 0) & (df["Customers"] != 0) & (df["is_used"] == 0 | df["is_used"] == 0.0 ))].iloc[:batch_size]
            else:
                result  =  df[(((df["Store"])== store_id) & (df["Sales"] != 0) & (df["Customers"] != 0) & ((df["is_used"] == 0) | (df["is_used"] == 0.0 )))].iloc[:batch_size]

            if(len(result) > 0):
                result['is_used'] = 1
                df.update(result)
                self.df.update(result)
                batches.append(result.index.values)

            else:
                isFinished = True

        # print(result.index.values)
        # print(batches)

        return batches

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

        ndays = self.days_between(start_date, end_date)
        arr_ndays = list(range(ndays))
        diff = random.choice(arr_ndays)
        res_result = pd.to_datetime(start_date) + pd.DateOffset(days=np.int(diff))
        orig_date = str(res_result)

        d = datetime.datetime.strptime(orig_date, '%Y-%m-%d %H:%M:%S')
        d = d.strftime('%Y-%m-%d')
        return d

        # return datetime.date.start_date + datetime.date(days = np.float(diff))

    def reset_dataset(self):
        self.df['is_used'] = 0
