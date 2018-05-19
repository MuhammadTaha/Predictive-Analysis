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
    def __init__(self, dir="data"):
        self.data_dir = dir

        # check if files are extracted
        if set(os.listdir()) >= set(["sample_submission.csv", "store.csv", "test.csv", "train.csv"]):
            print("Data is extracted already")
        else:
            Data.extract(dir+"/data.zip", dir)

        # load into pandas
        self.store_data = pd.read_csv(dir+"/store.csv")
        self.unlabeled = pd.read_csv(dir+"/test.csv")
        self.raw_train = pd.read_csv(dir+"/train.csv")
        pdb.set_trace()

        # split into train/test data


    @staticmethod
    def extract(path="data/data.zip", dest="data"):
        print("Extract from {}".format(path))
        zip_ref = zipfile.ZipFile(path, 'r')
        zip_ref.extractall(dest)
        zip_ref.close()