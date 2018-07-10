from sklearn import datasets
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from sklearn.datasets import dump_svmlight_file
import numpy as np
from sklearn.metrics import precision_score
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from forecaster.XGBForecaster import XGBForecaster
from data.feedforward_data import Data



def test2():
    try:
        data = Data.load_data()
    except:
        data = Data()
        data.save()
    try:
        forecaster = XGBForecaster.load_model("XGBForecaster2018-07-03-16:01")
    except:
        forecaster = XGBForecaster()
        forecaster.fit(data)
    print("validating model")
    forecaster.score(data)


if __name__ == '__main__':
    test2()
