try:
    from src.forecaster.abstract_forecaster import *
except ModuleNotFoundError:
    from .abstract_forecaster import *
from sklearn.svm import SVR
import numpy as np

class SVRForecaster(AbstractForecaster):
    params_grid = {
        "epsilon": np.linspace(0., 0.5, 20),
        "gamma": np.logspace(-8, 2, 20),
        "C": np.linspace(1, 5, 20)
    }

    def __init__(self, epsilon=5, gamma='auto', C=1.):
        self.svr = SVR(kernel='poly', C=1e3, degree=3)

    def _decision_function(self, X):
        X_norm = X/self.std
        return self.svr.predict(X_norm)

    def _train(self, X, y, **kwargs):
        self.std = np.std(X, axis=0, keepdims=True)
        self.std = np.where(self.std == 0, 1, self.std)
        X_norm = X / self.std
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