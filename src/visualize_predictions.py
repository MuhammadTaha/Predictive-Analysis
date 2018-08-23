try:
    from src.data.timeseries_data import TimeSeriesData
    from src.forecaster import *
except ModuleNotFoundError:
    from data.timeseries_data import TimeSeriesData
    from forecaster import *
import os
import numpy as np
from matplotlib import pyplot as plt

src_dir = os.path.dirname(os.path.abspath(__file__))


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
        date_error = forecaster.score(X, y)
        return np.mean(p), date_error

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


def visualize_predictions(forecaster, output_dir, data = None):
    """
    visualizes predictions for a forecaster
    :param forecaster: AbstractForecaster or str where to load a forecaster
    :param output_dir: str where to save the plots
    Visualizations:
    - Avg prediction and error per day
    - predictions and error for a random store
    """
    if data is None:
        data = TimeSeriesData()
    date_keys = sorted(data.train.Date.unique())

    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass

    # plot average sales + mse per day
    row_ids = list(range(1, data.time_count))
    plot_rows(data, forecaster, row_ids, "all", output_dir)

    # plot average per weekday
    for week_day, name in enumerate(["Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Monday"]):
        row_ids = data.train.index[data.train.Date.isin(date_keys[week_day::7])]
        plot_rows(data, forecaster, row_ids, week_day, output_dir)

    # plot random store
    store_id = np.random.randint(1, data.store_count)
    row_ids = data.train.index[data.train.Store == store_id]
    plot_rows(data=data, forecaster=forecaster, row_ids=row_ids, name="Store-{}".format(store_id), output_dir=output_dir)


def visualize_predictions_quick(forecaster, output_dir, store_id):
    """
    visualizes predictions for a forecaster
    :param forecaster: AbstractForecaster or str where to load a forecaster
    :param output_dir: str where to save the plots
    Visualizations:
    - Avg prediction and error per day
    - predictions and error for a random store
    """
    data = TimeSeriesData(keep_zero_sales=True)
    date_keys = sorted(data.train.Date.unique())

    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass
    print("plot to ", output_dir)

    row_ids = data.train.index[data.train.Store == store_id]
    plot_rows(data=data, forecaster=forecaster, row_ids=row_ids, name="Store-{}".format(store_id), output_dir=output_dir)