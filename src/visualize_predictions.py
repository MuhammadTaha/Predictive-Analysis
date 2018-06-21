from data import *
import pdb
import tensorflow as tf
from forecaster import *
import os
import numpy as np

src_dir = os.path.dirname(os.path.abspath(__file__))

def time_series_plots(days, date_keys, sales, predictios, name):
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

def plot_rows(data, forecaster, row_ids, name, output_dir):
    """
    plot avg sales, avg predicted sales and mse per day
    """
    date_keys = sorted(data.train.iloc[row_ids].Date.unique())
    dates = [data.str_to_date(i) for i in date_keys]
    days = [(i - dates[0]).days for i in dates]

    def sales_for(date):
        return np.array(data.train.loc[data.train["Date"] == date]["Sales"])

    def predictions_for(date):
        X, y = data._extract_rows(data.train.index[data.train["Date"] == date].tolist())
        p = forecaster.predict(X)
        print("predicted: ", p)
        pdb.set_trace()
        error = forecaster.sess.run(forecaster.loss, feed_dict = {forecaster.input: X, forecaster.true_sales: y})
        return np.mean(p), error


    avg_sales = [np.mean(sales_for(i)) for i in date_keys]
    A = np.array([predictions_for(i) for i in date_keys])
    avg_predictions = A[:, 0]
    error = A[:, 1]

    plt.plot(days, avg_sales, label="Sales")
    plt.plot(days, avg_predictions, label="Predictions")
    plt.legend()
    plt.xticks([days[0], days[-1]], [date_keys[0], date_keys[-1]])
    plt.savefig(output_dir+"/{}.png".format(name))
    plt.clf()

    plt.plot(days, error)
    plt.ylabel("Error")
    plt.xticks([days[0], days[-1]], [date_keys[0], date_keys[-1]])
    plt.savefig(output_dir + "/{}-error.png".format(name))


def visualize_predictions(forecaster, output_dir):
    """
    visualizes predictions for a forecaster
    :param forecaster: AbstractForecaster or str where to load a forecaster
    :param output_dir: str where to save the plots
    Visualizations:
    - Avg prediction per day
    - predictions for some random stores
    """
    data = TimeSeriesData()
    date_keys = sorted(data.train.Date.unique())

    try: os.makedirs(output_dir)
    except FileExistsError: pass

    plot_rows(data, forecaster, list(range(1, 20000, 1000)), "test", output_dir)
    # plot average sales + mse per day
    row_ids = list(range(1, data.time_count))
    plot_rows(data, forecaster, row_ids, "all", output_dir)

    # plot average per weekday
    for week_day, name in enumerate(["Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Monday"]):
        row_ids = data.train.index[data.train.Date in date_keys[week_day::7]]
        plot_rows(data, forecaster, row_ids, week_day, output_dir)

    # plot random store
    store_id = np.random.randint(1, data.store_count)
    row_ids = data.train.index[data.Store == store_id]
    plot_rows(data, forecaster, row_ids, "Store-{}".format(store_id), output_dir)

