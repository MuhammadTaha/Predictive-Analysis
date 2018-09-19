try:
    from src.forecaster.abstract_forecaster import *
except ModuleNotFoundError:
    from .abstract_forecaster import *
from sklearn.svm import SVR
import numpy as np

class SVRForecaster(AbstractForecaster):
    params_grid = {
        "epsilon": list(range(0, 20, 2)),
        "gamma": ['auto'] + list(range(1, 100, 10))
    }

    def __init__(self, epsilon=5, gamma='auto'):
        self.svr = SVR(epsilon=epsilon, gamma=gamma)

    def _decision_function(self, X):
        return self.svr.predict(X)

    def _train(self, data):
        X, y = data.all_train_data()
        self.svr.fit(X, y)

    def _build(self):
        pass

    def score(self, X, y):
        prediction = self.predict(X)
        return rmspe(y, prediction)

    def save(self):
        file_name = self.__class__.__name__ + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
        params = self.svr.get_params()
        joblib.dump(params, os.path.join(MODEL_DIR, file_name))
        return os.path.join(MODEL_DIR, file_name)

    @staticmethod
    def load_model(file_name):
        model = SVRForecaster()
        model.svr.set_params(joblib.load(os.path.join(MODEL_DIR, file_name)))
        return model