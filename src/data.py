import pandas as pd
import zipfile
import os
import tensorflow as tf
import numpy as np

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
    def __init__(self, dir="data", p_train=0.6, p_val=0.2, p_test=0.2):
        assert p_train+p_val+p_test == 1
        
        self.data_dir = dir

        # check if files are extracted
        if set(os.listdir()) >= set(["sample_submission.csv", "store.csv", "test.csv", "train.csv"]):
            print("Data is extracted already")
        else:
            Data.extract(dir+"/data.zip", dir)

        # load into pandas
        self.store = pd.read_csv(dir+"/store.csv")
        self.final_test = pd.read_csv(dir+"/test.csv")
        self.train = pd.read_csv(dir+"/train.csv")

        self.time_count = self.train.shape[0]
        self.store_count = self.store.shape[0]
        pdb.set_trace()

        # split into train/test data
        self.train_ids = np.arange()

    def next_train_batch(self, batch_size=50):
        pass


    @staticmethod
    def extract(path="data/data.zip", dest="data"):
        print("Extract from {}".format(path))
        zip_ref = zipfile.ZipFile(path, 'r')
        zip_ref.extractall(dest)
        zip_ref.close()