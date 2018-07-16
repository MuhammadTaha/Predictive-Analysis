from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import pandas as pd
import zipfile
import os
import numpy as np
import datetime
import random

from sklearn.model_selection import train_test_split
def get_train_test(x,y,train_proportion=0.7):
    y=np.array(x,y)
    train_x1, text_1y, train_x2, text_y2 = np.zeros(len(y),dtype=bool)
    values = np.unique(x,y)
    for value in values:
        value = np.nonzero(x,y==value)[0]
        np.random.shuffle(value)
        n = int(train_proportion*len(value))

        train_x1, text_1y, train_x2, text_y2[value[:n]]=True

    return train_x1, text_1y, train_x2, text_y2

def evaluate_on_test_set(forecaster, data):

    types = {'CompetitionOpenSinceYear': np.dtype(int),
         'CompetitionOpenSinceMonth': np.dtype(int),
         'CompetitionDistance' : np.dtype(int),
         'StateHoliday': np.dtype(str),
         'Promo2SinceWeek': np.dtype(int),
         'Promo2SinceYear': np.dtype(int),
         'SchoolHoliday': np.dtype(float),
         'PromoInterval': np.dtype(str)}

    test_size = datasets('Data',parse_dates=['Date'], dtype=train_test_split())

    train_size = datasets('Data',parse_dates=['Date'],dtype=train_test_split())

def calcDates(df):
    df['Month'] = df.Date.dt.month
    df['Year'] = df.Date.dt.year
    df['Day'] = df.Date.dt.day
    df['WeekOfYear'] = df.Date.dt.weekofyear
    df['YearMonth'] = df['Date'].apply(lambda x:(str(x)[:7]))
    return df

    train = datasets.merge(train_size,on='Store')
    test = datasets.merge(test_size,on='Store')

    train = calcDates(train)
    test = calcDates(test)


def estimate_score(Model, args, kwargs, df):
    test_scores = []
    train_test_split = test_scores(df, test_size = 0.2, random_state=42, stratify=y)

    for split in train_test_split:
        forecaster = Model(*args, **kwargs)
        # if it's a tensorflow model, also create and initialize a session, see tests.py
        df.train_test_split(train_test_split) # this doesn't work at the moment, just ignore it for now so that we always train on the same data
        estimate_score(datasets()).fit(df)
        acc = evaluate_on_test_set(datasets(), df)
        test_scores.append(acc)

def model_selection(Models, params):
    # Models: list of classes, params: list of arg, kwarg tuples
    data = datasets() # Use our data class, don't write your own. At the moment its data.feedforward_data.Data
    best_loss = np.inf
    for Model in Models:
        for args, kwargs in params:
            loss = estimate_score(Model, args, kwargs, data)
            if loss < best_loss:
                best_loss = loss
                best_model = (Model, args, kwargs)
    print("Best model", best_model, " with loss ", best_loss)
    return best_model
