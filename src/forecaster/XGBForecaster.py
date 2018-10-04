# from sklearn.cross_validation import train_test_split
import xgboost as xgb

from src.data import FeedForwardData

try:
    from src.forecaster.abstract_forecaster import *  # we need more than AbstractForecaster, don't change it to only import that
except ModuleNotFoundError:
    print("Use relative import without src")
    from .abstract_forecaster import *


class XGBForecaster(AbstractForecaster):
    def _build(self):
        pass

    params_grid = {
        'n_estimators': np.linspace(500, 1000, num=5).astype(int),
        'max_depth': np.linspace(10, 20, num=5).astype(int),
        "eta": np.linspace(0.1, 0.3, num=5),
        "objective": ["gpu:reg:linear"],
        "booster": ["gbtree"],
        "subsample": np.linspace(0.7, 1.0, 5),
        "colsample_bytree": [0.7],
        "silent": [1]
    }

    # initial_params = {
    #     'n_estimators': 800,
    #     "learning_rate": .2,
    #     'objective': "gpu:reg:linear",
    #     'silent': True,
    #     "tree_method": "gpu_hist"
    # }
    #
    early_stopping_rounds = 100

    def __init__(self, **kwargs):
        super().__init__()
        self.params = kwargs

    def _train(self, data, **kwargs):
        X, y = data.all_train_data()
        split = int(len(X) * 0.05)
        X_train, y_train = data.all_train_data()
        dtrain = xgb.DMatrix(X_train[split:], y_train[split:])
        dtest = xgb.DMatrix(X_train[:split], y_train[:split])
        watchlist = [(dtrain, 'train'), (dtest, 'eval')]
        self.model = xgb.train(self.params, dtrain, kwargs['n_rounds'], evals=watchlist, early_stopping_rounds=100,
                               feval=self.rmspe_xg)
        # return self.model.__dict__

    def save(self, trial):
        file_name = "{}-{}".format(self.__class__.__name__, trial)
        self.model.save_model(os.path.join(MODEL_DIR, file_name))

    # def load_model(self, file_name):
    #     self.model = xgb.load

    def _decision_function(self, X):
        return self.model.predict(xgb.DMatrix(X))

    def score(self, X, y):
        predictions = self._decision_function(X)
        error = self.rmspe(y, predictions)
        print("Validation score: {:.4f}".format(error))
        return error

    # def _search_hyper_params(self, X_train, y_train):
    #     initial_model = xgb.XGBRegressor(**self.initial_params)
    #     search_model = RandomizedSearchCV(initial_model, self.params_grid, cv=6)
    #     search_model = search_model.fit(X_train, y_train)
    #     return {**search_model.best_params_, **self.initial_params}

    @staticmethod
    def ToWeight(y):
        w = np.zeros(y.shape, dtype=float)
        ind = y != 0
        w[ind] = 1. / (y[ind] ** 2)
        return w

    @staticmethod
    def rmspe(yhat, y):
        w = XGBForecaster.ToWeight(y)
        rmspe = np.sqrt(np.mean(w * (y - yhat) ** 2))
        return rmspe

    @staticmethod
    def rmspe_xg(yhat, y):
        # y = y.values
        y = y.get_label()
        y = np.exp(y) - 1
        yhat = np.exp(yhat) - 1
        w = XGBForecaster.ToWeight(y)
        rmspe = np.sqrt(np.mean(w * (y - yhat) ** 2))
        return "rmspe", rmspe

    @staticmethod
    def load_model(file_name):
        model = xgb.Booster({})  # init model
        model.load_model(file_name)
        return model


if __name__ == '__main__':
    data = FeedForwardData()
    points = list(range(2000))
    points = np.random.permutation(points)
    split = int(0.7 * len(points))
    data.train_test_split(set(points[:split]), set(points[split:]))
    forecaster = XGBForecaster()
    forecaster._train(data)
