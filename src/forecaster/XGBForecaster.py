# from sklearn.cross_validation import train_test_split
import sklearn as sk
import xgboost as xgb

from src.data import FeedForwardData

try:
    from src.forecaster.abstract_forecaster import *  # we need more than AbstractForecaster, don't change it to only import that
except ModuleNotFoundError:
    print("Use relative import without src")
    from .abstract_forecaster import *


class XGBForecaster(AbstractForecaster, sk.base.BaseEstimator):
    def _build(self):
        pass

    params_grid = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
    }
    # params_grid = {
    #     'learning_rate': np.random.uniform(0.01, 0.3, 2),
    #     'max_depth': list(range(10, 20, 2)),
    #     'gamma': np.random.uniform(0, 10, 2),
    #     'reg_alpha': np.random.exponential(1, 10)}

    initial_params = {
        'n_estimators': 800,
        "learning_rate": .2,
        'max_depth': 12,
        'objective': "reg:logistic",
        'silent': True,
    }

    n_rounds = 3000
    early_stopping_rounds = 100

    def __init__(self, load=False, **kwargs):
        if 'params' in kwargs:
            self.params = kwargs['params']
        else:
            self.params = self.initial_params
        if load:
            self.model = xgb.Booster(self.params)  # init model
            self.model.load_model(os.path.join(MODEL_DIR, kwargs['file_path']))

    def _train(self, data):
        X, y = data.all_train_data()
        # X_train, X_test = train_test_split(X, test_size=0.05)
        split = int(len(X) * 0.05)
        X_train, y_train = data.all_train_data()
        dtrain = xgb.DMatrix(X_train[split:], y_train[split:])
        dtest = xgb.DMatrix(X_train[:split], y_train[:split])
        watchlist = [(dtrain, 'train'), (dtest, 'eval')]
        self.model = xgb.train(self.params, dtrain, self.n_rounds, evals=watchlist, early_stopping_rounds=100,
                               feval=self.rmspe_xg)
        self.save()

    def save(self):
        file_name = self.__class__.__name__ + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
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
    def rmspe(y, yhat):
        return rmspe(y, yhat)

    @staticmethod
    def rmspe_xg(yhat, y):
        y = np.expm1(y.get_label())
        yhat = np.expm1(yhat)
        return "rmspe", XGBForecaster.rmspe(y, yhat)


if __name__ == '__main__':
    data = FeedForwardData()
    points = list(range(2000))
    points = np.random.permutation(points)
    split = int(0.7 * len(points))
    data.train_test_split(set(points[:split]), set(points[split:]))
    forecaster = XGBForecaster()
    forecaster._train(data)
