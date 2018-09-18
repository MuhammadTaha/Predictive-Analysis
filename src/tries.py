"""
This is for tries that should not ent up in the actual app
"""

from visualize_predictions import visualize_predictions
import pdb
import tensorflow as tf
from forecaster import *
from data import *

try:
    from src.data import *
    from src.forecaster import *
except ModuleNotFoundError:
    print("Use relative import without src")
    from data import *
    from forecaster import *


src_dir = os.path.dirname(os.path.abspath(__file__))


def howmanyfeatures():
    print("We currently have {} features".format(Data().features_count))


def linear_regression(train_new=False):
    with tf.Session() as sess:
        data = Data(toy=True)
        model = LinearRegressor(sess=sess,
                                plot_dir=src_dir + "/../plots/linear-regression",
                                features_count=data.features_count)

        try:
            assert not train_new
            model.load_params("models/LinearRegressor2018-06-28-22:55")
        except (tf.errors.NotFoundError, AssertionError) as e:
            sess.run(tf.global_variables_initializer())
            model.fit(data)
            print("Save model to ", model.save())

        visualize_predictions(model, src_dir + "/../plots/linear-regression")


def feedforwardnn(train_new=False):
    with tf.Session() as sess:
        data = Data(toy=True)
        model = FeedForwardNN1(sess=sess,
                               plot_dir=src_dir + "/../plots/feed-forward-nn",
                               features_count=data.features_count)
        try:
            assert not train_new
            model.load_params("models/LinearRegressor2018-06-28-22:55")
        except (tf.errors.NotFoundError, AssertionError) as e:
            if not train_new:
                print(e)
                print("Starting new training")
            sess.run(tf.global_variables_initializer())
            model.fit(data)
            print("Save model to ", model.save())

        visualize_predictions(model, src_dir + "/../plots/feed-forward-nn")


def time_series_example():
    # how to use time series data
    data = TimeSeriesData()
    for store in range(1115):
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


def naive_classifier():
    data = Data(toy=True)
    print("y shape of data", data.y_val.shape)
    model = NaiveForecaster()
    print("Naive forecaster score:", model.score(data.X_val, data.y_val))
    p = model.predict(data.X_val)
    diff = np.array([p, data.y_val])
    print("diff", diff.shape)
    pdb.set_trace()
    visualize_predictions(model, src_dir + "/../plots/naive_forecaster")


def try_tf_model(model_type, plot_dir=None, load_from=None):
    if plot_dir is not None:
        plot_dir = src_dir + "/../plots/{}".format(model_type.__name__)

    with tf.Session() as sess:
        data = Data(toy=True)
        model = model_type(sess=sess,
                           plot_dir=plot_dir,
                           features_count=data.features_count)
        try:
            assert load_from is not None
            model.load_params(load_from)
        except (tf.errors.NotFoundError, AssertionError) as e:
            if load_from is not None:
                print(e)
                print("Starting new training")
            sess.run(tf.global_variables_initializer())
            model.fit(data)
            print("Save model to ", model.save())

        visualize_predictions(model, plot_dir)


def try_model_wo_sess(model_type, plot_dir=None):
    if plot_dir is not None:
        plot_dir = src_dir + "/../plots/{}".format(model_type.__name__)

    data = Data(toy=True)
    model = model_type()

    model.fit(data)
    print("score:", model.score(data.X_val, data.y_val))
    p = model.predict(data.X_val)
    diff = np.array([p, data.y_val])[:,:,0].T
    print("diff", diff.shape)

    visualize_predictions(model, plot_dir)


def test_lstm_data():
    data = LSTMData(is_debug=True, update_disk=True)
    data.train_test_split(list(range(500)), list(range(500, 750)))
    X, Y = data.next_train_batch()
    print("one train: X, Y: ", np.array(X).shape, np.array(Y).shape)
    X, Y = data.all_test_data()
    print("all test: X, Y: ", np.array(X).shape, np.array(Y).shape)

    lstm = LSTMForecaster(num_timesteps=data.num_tsteps, features_count=data.features_count)
    lstm.fit(data)
    pdb.set_trace()
    print(data.X_val)
    print(lstm.score(data.X_val, data.y_val))



def main():
    test_lstm_data()
    #  d = Data()
    #  choose the methods to try here
    #  time_series_example()
    #  howmanyfeatures()
    #  test_order_of_dates()
    #  time_series_example()
    #
    #  linear_regression(train_new=False)
    #  AbstractData.next_train_batch(AbstractData,store_id = 2,forecaster = "linear regressor" , batch_size= 10)
    #  feedforwardnn(train_new=True)
    #  naive_classifier()
    #  try_model_wo_sess(NaiveForecaster2)

