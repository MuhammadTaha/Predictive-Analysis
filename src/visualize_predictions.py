from data import *
import pdb
import tensorflow as tf
from forecaster import *
import os

src_dir = os.path.dirname(os.path.abspath(__file__))

def visualize_predictions(forecaster, output_dir):
    """
    visualizes predictions for a forecaster
    :param forecaster: AbstractForecaster or str where to load a forecaster
    :param data: Data object
    :param output_dir: str where to save the plots
    Visualizations:
    - Avg prediction per day
    - predictions for some random stores
    """
    data = TimeSeriesData()
    predicted = PredictedTimeseriesData(data, forecaster)


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
    pass
