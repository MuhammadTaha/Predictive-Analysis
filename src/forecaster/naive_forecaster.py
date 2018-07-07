from .abstract_forecaster import *


class NaiveForecaster(AbstractForecaster):
    """
    Return the average of the store on the given weekday
    """
    def __init__(self):
        self.trained = True

    def _decision_function(self, X):
        return X[:, FEATURES["weekday_store_avg"], None]*X[:, FEATURES["open"], None]

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
        return X[:, FEATURES["week_of_year_avg"], None]*X[:, FEATURES["open"], None]

    def _train(self, *args, **kwargs):
        return

    def _build(self):
        pass

    def score(self, X, y):
        prediction = self.predict(X)
        return np.sqrt(np.mean(np.square((prediction - y + 0.1) / (y + 0.1))))
