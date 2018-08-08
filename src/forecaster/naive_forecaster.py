try:
    from src.forecaster.abstract_forecaster import *
except ModuleNotFoundError:
    from .abstract_forecaster import *
import numpy as np


class NaiveForecaster(AbstractForecaster):
    """
    Return the average of the store on the given weekday
    """
    def __init__(self):
        self.trained = True

    def _decision_function(self, X):
        return X[:, WEEKDAY_STORE_AVG, None]*X[:, OPEN, None]

    def _train(self, *args, **kwargs):
        return

    def _build(self):
        pass

    def score(self, X, y):
        prediction = self.predict(X)
        return np.sqrt(np.mean(np.square((prediction - y + 0.1) / (y + 0.1))))


class NaiveForecaster2(AbstractForecaster):
    """
    Return the average of the store in the week of the year
    """
    def __init__(self):
        self.trained = True

    def _decision_function(self, X):
        return X[:, WEEK_OF_YEAR_AVG, None]*X[:, OPEN, None]

    def _train(self, *args, **kwargs):
        return

    def _build(self):
        pass

    def score(self, X, y):
        prediction = self.predict(X)
        return np.sqrt(np.mean(np.square((prediction - y + 0.1) / (y + 0.1))))
