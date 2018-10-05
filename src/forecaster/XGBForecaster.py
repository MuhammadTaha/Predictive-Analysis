# from sklearn.cross_validation import train_test_split
import xgboost as xgb

try:
    from src.forecaster.abstract_forecaster import *  # we need more than AbstractForecaster, don't change it to only import that
except ModuleNotFoundError:
    print("Use relative import without src")
    from .abstract_forecaster import *


class XGBForecaster(AbstractForecaster):
    def _build(self):
        pass

    params_grid = {
        'n_estimators': np.linspace(500, 700, num=3).astype(int),
        'max_depth': np.linspace(10, 15, num=3).astype(int),
        "eta": [0.2],
        "objective": ["gpu:reg:linear"],
        "booster": ["gbtree"],
        "subsample": np.linspace(0.8, 0.9, 3),
        "colsample_bytree": [0.7],
        "silent": [1]
    }

    params = {"objective": "gpu:reg:linear",
              "booster": "gbtree",
              "eta": 0.2,
              "max_depth": 12,
              "subsample": 0.9,
              "colsample_bytree": 0.7,
              "silent": 1
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

    def _train(self, X, y, **kwargs):
        split = int(len(X) * 0.05)
        dtrain = xgb.DMatrix(X[split:], y[split:])
        dtest = xgb.DMatrix(X[:split], y[:split])
        watchlist = [(dtrain, 'train'), (dtest, 'eval')]
        self.model = xgb.train(self.params, dtrain, kwargs['n_rounds'], evals=watchlist, early_stopping_rounds=100,
                               feval=self.loss, )
        # return self.model.__dict__

    def save(self, trial):
        file_name = "{}-{}".format(self.__class__.__name__, trial)
        self.model.save_model(os.path.join(MODEL_DIR, file_name))

    # def load_model(self, file_name):
    #     self.model = xgb.load

    def _decision_function(self, X):
        return self.model.predict(xgb.DMatrix(X))

    # def score(self, X, y):
    #     predictions = self._decision_function(X)
    #     error = self.rmspe(y, predictions)
    #     print("Validation score: {:.4f}".format(error))
    #     return error

    # def _search_hyper_params(self, X_train, y_train):
    #     initial_model = xgb.XGBRegressor(**self.initial_params)
    #     search_model = RandomizedSearchCV(initial_model, self.params_grid, cv=6)
    #     search_model = search_model.fit(X_train, y_train)
    #     return {**search_model.best_params_, **self.initial_params}

    @staticmethod
    def loss(yhat, y):
        y = y.get_label()
        error = np.mean(np.absolute(yhat - y))
        return "mae", error

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
