try:
    from src.forecaster.abstract_forecaster import *
except ModuleNotFoundError:
    from .abstract_forecaster import *
from sklearn.svm import SVR
import numpy as np

class SVRForecaster(AbstractForecaster):
    params_grid = {
        "epsilon": np.logspace(1, 2, 20),
        "gamma": np.logspace(-5, 5, 20),
        "C": np.linspace(1, 1000, 20)
    }

    def __init__(self, epsilon=5, gamma='auto', C=1.):
        self.svr = SVR(epsilon=epsilon, gamma=gamma, C=C)

    def _decision_function(self, X):
        return self.svr.predict(X)

    def _train(self, data):
        X, y = data.all_train_data()

        self.svr.fit(X, y)

    def fit(self, data):
        X, y = data.all_train_data()
        X_norm = X/np.std(X, axis=1, keepdims=True)
        self.svr.fit(X_norm, y)

    def _build(self):
        pass

    def score(self, X, y):
        prediction = self.predict(X)
        return rmspe(y, prediction)

    def save(self):
        file_name = self.__class__.__name__ + datetime.datetime.now().strftime("%Y-%m-%d-%H")
        params = self.svr.get_params()
        joblib.dump(params, os.path.join(MODEL_DIR, file_name))
        return os.path.join(MODEL_DIR, file_name)

    @staticmethod
    def load_model(file_name):
        model = SVRForecaster()
        model.svr.set_params(joblib.load(os.path.join(MODEL_DIR, file_name)))
        return model