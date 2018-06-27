# from sklearn.cross_validation import train_test_split

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from src.forecaster import AbstractForecaster
import xgboost as xgb
import numpy as np


class XGBForecaster(AbstractForecaster):
    params_grid = {
        'learning_rate': np.random.uniform(0.01, 0.3, 2),
        'max_depth': list(range(10, 20, 2)),
        'gamma': np.random.uniform(0, 10, 2),
        'reg_alpha': np.random.exponential(1, 10)}

    initial_params = {
        'n_estimators': 500,
        'objective': 'reg:linear',
        'subsample': 0.8,
        'colsample_bytree': 0.85,
        'seed': 42,
        'silent': True,
    }

    n_rounds = 500
    early_stopping_rounds = 50

    def __init__(self, **kwargs):
        if 'params' in kwargs:
            self.params = kwargs['params']
        else:
            self.params = None

    def _build(self):

        pass

    def _train(self, X, y):
        # X, y = self.clean_features(data)
        if self.params == None:
            self._search_hyper_params(X, y)
        # print(self.params)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)
        dtrain = xgb.DMatrix(X_train, y_train)
        dtest = xgb.DMatrix(X_test, y_test)
        watchlist = [(dtrain, 'train'), (dtest, 'test')]
        self.model = xgb.train(self.params, dtrain, self.n_rounds, evals=watchlist, feval=self.rmspe_xg,
                               early_stopping_rounds=self.early_stopping_rounds)

    def clean_features(self, data):

        return data.X_val, data.y_val

    def _decision_function(self, X):
        return self.model.predict(xgb.DMatrix(X))

    def score(self, X, y=None):
        predictions = self._decision_function(X)
        error = self.rmspe(y, predictions)
        print("Validtaion score: {:.4f}".format(error))
        return error

    def _search_hyper_params(self, X_train, y_train):
        
        initial_model = xgb.XGBRegressor(**self.initial_params)
        search_model = RandomizedSearchCV(initial_model, self.params_grid, cv=6)
        search_model = search_model.fit(X_train[:300], y_train[:300])
        print(search_model.best_params_)
        print(search_model.best_score_)
        self.params = {**search_model.best_params_, **self.initial_params}

    @staticmethod
    def rmspe(y, yhat):
        return np.sqrt(np.mean((yhat / y - 1) ** 2))

    @staticmethod
    def rmspe_xg(yhat, y):
        y = np.expm1(y.get_label())
        yhat = np.expm1(yhat)
        return "rmspe", XGBForecaster.rmspe(y, yhat)
