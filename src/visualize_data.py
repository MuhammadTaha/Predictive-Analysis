from matplotlib import pyplot as plt
import numpy as np
import os
from sklearn.svm import SVR
import tensorflow as tf

try:
    from src.data.feedforward_data import Data
    from src.data.predicted_data import PredictedTimeseriesData
    from src.forecaster import LinearRegressor
except ModuleNotFoundError:
    print("Use relative import without src")
    from data.feedforward_data import Data
    from data.predicted_data import PredictedTimeseriesData
    from forecaster import LinearRegressor

import pdb

src_dir = os.path.dirname(os.path.abspath(__file__))

def smooth(vals, gamma=1):
    X = np.array(range(len(vals)))[..., None]
    svr = SVR(gamma=3).fit(X, vals)
    return svr.predict(X)

def _time_series_plots(days, date_keys, sales, name):
    target_dir = src_dir + "/../plots/" + name
    try: os.makedirs(target_dir)
    except FileExistsError: pass

    num_sales = [len(i) for i in sales]
    avg_sales = np.array([np.mean(i) for i in sales])
    std_sales = np.array([np.std(i) for i in sales])
    lower = avg_sales - std_sales
    upper = avg_sales + std_sales

    # pdb.set_trace()

    # how many entries per day?
    plt.plot(days, num_sales)
    plt.xticks([days[0], days[-1]], [date_keys[0], date_keys[-1]])
    plt.ylabel("Stores with data")
    plt.savefig(target_dir + "/data_per_day.png")
    plt.clf()

    # sales per day
    plt.plot(days, avg_sales, color="blue")
    plt.plot(days, lower, "--", color="green")
    plt.plot(days, upper, "--", color="green")
    plt.xticks([days[0], days[-1]], [date_keys[0], date_keys[-1]])
    plt.ylabel("Sales")
    plt.savefig(target_dir + "/sales_per_day.png")
    plt.clf()

    # smoothed sales per day
    plt.plot(days, smooth(avg_sales), color="blue")
    plt.plot(days, smooth(lower), "--", color="green")
    plt.plot(days, smooth(upper), "--", color="green")
    plt.xticks([days[0], days[-1]], [date_keys[0], date_keys[-1]])
    plt.ylabel("Sales")
    plt.savefig(target_dir + "/sales_per_day_smoothed.png")
    plt.clf()

    plt.plot(days, avg_sales, color="blue")
    plt.plot(days, smooth(lower, gamma=5), "--", color="green")
    plt.plot(days, smooth(upper, gamma=5), "--", color="green")
    plt.xticks([days[0], days[-1]], [date_keys[0], date_keys[-1]])
    plt.ylabel("Sales")
    plt.savefig(target_dir + "/sales_per_day_smoothed_std.png")
    plt.clf()

    # histogramm of sales per day
    plt.hist(sales)
    plt.xlabel("Sales")
    plt.savefig(target_dir + "/sales_hist.png")
    plt.clf()


def visualize_time_series():
    data = Data()
    # for each date: show avg sales + variance
    # for each date: show how many rows we have
    date_keys = sorted(data.train.Date.unique())
    dates = [data.str_to_date(i) for i in date_keys]
    days = [(i - dates[0]).days for i in dates]
    print("We have data for each date between {} and {} : {}".format(date_keys[0], date_keys[-1], days==list(range(len(days)))))

    def sales_for(date):
        return np.array(data.train.loc[data.train["Date"] == date]["Sales"])

    sales = [sales_for(i) for i in date_keys]
    _time_series_plots(days, date_keys, sales, "all")

    # plot of single week days
    def mean(wd_sales):
        return np.mean([np.mean(i) for i in wd_sales])

    wd_avg = []
    for week_day, name in enumerate(["Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Monday"]):
        wd_sales = sales[week_day::7]
        _time_series_plots(days[week_day::7], date_keys[week_day::7], wd_sales, name)
        wd_avg.append(mean(wd_sales))
    wd_avg = wd_avg[-1:] + wd_avg[:-1] # so that it starts with monday

    plt.bar(range(7), wd_avg)
    plt.xticks(range(7), ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"])
    plt.ylabel("Average Sales")
    plt.savefig(src_dir+"/../plots/weekdays.png")


def visualize_linear_dependencies():
    # train a linear model and plot the weights
    # visualize time series of predictions
    # scatter plot the most interesting features (with the highest weight)
    with tf.Session() as sess:
        data = Data()
        model = LinearRegressor(sess=sess,
                                plot_dir=src_dir+"/../plots/linear-regression",
                                features_count=data.features_count)
        sess.run(tf.global_variables_initializer())

        try:
            model.fit(data)
        except Exception as e:
            print(type(e), e)
            pdb.set_trace()
            model.fit(data)

        model.save()
        pdb.set_trace()


def visualize_predictions(forecaster, data, output_dir):
    """
    visualizes predictions for a forecaster
    :param forecaster: AbstractForecaster or str where to load a forecaster
    :param data: Data object
    :param output_dir: str where to save the plots
    Visualizations:
    - Avg prediction per day
    - predictions for some random stores
    """
    predicted = PredictedTimeseriesData(data, forecaster)

    data = Data(is_time_series=True)
    # for each date: show avg sales + variance
    # for each date: show how many rows we have
    date_keys = sorted(data.train.Date.unique())
    dates = [data.str_to_date(i) for i in date_keys]
    days = [(i - dates[0]).days for i in dates]
    print("We have data for each date between {} and {} : {}".format(date_keys[0], date_keys[-1],
                                                                     days == list(range(len(days)))))

    def sales_for(date):
        return np.array(data.train.loc[data.train["Date"] == date]["Sales"])

    sales = [sales_for(i) for i in date_keys]
    _time_series_plots(days, date_keys, sales, "all")

    # plot of single week days
    def mean(wd_sales):
        return np.mean([np.mean(i) for i in wd_sales])

    wd_avg = []
    for week_day, name in enumerate(["Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Monday"]):
        wd_sales = sales[week_day::7]
        _time_series_plots(days[week_day::7], date_keys[week_day::7], wd_sales, name)
        wd_avg.append(mean(wd_sales))
    wd_avg = wd_avg[-1:] + wd_avg[:-1]  # so that it starts with monday

    plt.bar(range(7), wd_avg)
    plt.xticks(range(7), ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"])
    plt.ylabel("Average Sales")
    plt.savefig(src_dir + "/../plots/weekdays.png")



def main():
    visualize_time_series()
    #visualize_linear_dependencies()