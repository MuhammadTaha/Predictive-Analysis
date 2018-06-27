from src.forecaster import AbstractForecaster
from xgboost import XGBClassifier

class XGBForecaster(AbstractForecaster):
    def __init__(self, **kwargs):
        pass

    def _train(self, X, y=None):
        pass

    def _decision_function(self, X):
        pass

    def _build(self):
        pass

    def score(self, X, y=None):
        pass
