"""
This is for tries that should not ent up in the actual app
"""
from data import *
from visualize_predictions import visualize_predictions
import pdb
import tensorflow as tf
from forecaster import *
import os
"""
Todo:
    - plot predictions
        - 
    - next batch schreiben für time series, das kleine stücke einer time series nimmt
"""
src_dir = os.path.dirname(os.path.abspath(__file__))


def howmanyfeatures():
    print("We currently have {} features".format(Data().features_count))


def linear_regression(train_new=False):
    with tf.Session() as sess:
        model = LinearRegressor(sess=sess,
                                plot_dir=src_dir + "/../plots/linear-regression",
                                features_count=25)

        try:
            assert not train_new
            model.load_params("models/LinearRegressor2018-06-21-20:31_params")
        except (tf.errors.NotFoundError, AssertionError) as e:
            data = Data()
            sess.run(tf.global_variables_initializer())
            model.fit(data)
            print("Save model to ", model.save())

        visualize_predictions(model, src_dir + "/../plots/linear-regression")


def feedforwardnn(train_new=False):
    with tf.Session() as sess:
        model = FeedForwardNN1(sess=sess,
                                plot_dir=src_dir + "/../plots/feed-forward-nn",
                                features_count=25)
        try:
            assert not train_new
            model.load_params("models/FeedForwardNN12018-06-21-20:31_params")
        except (tf.errors.NotFoundError, AssertionError) as e:
            data = Data()
            sess.run(tf.global_variables_initializer())
            model.fit(data)
            print("Save model to ", model.save())

        visualize_predictions(model, src_dir + "/../plots/feed-forward-nn")


def time_series_example():
    # how to use time series data
    data = TimeSeriesData()
    for store in range(1119):
        days, _, y = data._get_time_series(store)
        print("Labels for store {}: {}".format(store, y.shape))
        print("Day index for each row", days)


def test_order_of_dates():
    data = TimeSeriesData()
    row_ids = data.train.index[data.train.Store == 5].tolist()[::-1]
    print("First date: ", data.train.loc[row_ids[0]]["Date"])
    print("Last date: ", data.train.loc[row_ids[-1]]["Date"])
    days = [data.str_to_date(d) for d in data.train.loc[row_ids]["Date"]]

    days = [(d - days[0]).days for d in days]
    print("Date steps", days)


def main():
    #  choose the methods to try here
    #  time_series_example()
    #  howmanyfeatures()
    #  test_order_of_dates()
    #  time_series_example()
    #
    #linear_regression(train_new=True)
    feedforwardnn(train_new=True)
