import numpy as np
from sklearn import cross_validation
import xgboost as xgb
import os
import numpy as np

src_dir = os.path.dirname(os.path.abspath(__file__))

def weight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1. / (y[ind] ** 2)
    return w


def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat) ** 2))
    return rmspe


def rmspe_xg(yhat, y):
    # y = y.values
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat) ** 2))
    return "rmspe", rmspe


# Gather some features
def build_features(features, data):
    # remove NaNs
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), 'Open'] = 1
    # Use some properties directly
    features.extend(
        ('store_type', 'assortment', ['competition_distance', 'competition_open', 'promo_since_days',
                                  'days_since_interval'], 'day_of_week', '[open, promo]', 'state_holiday',
         '[school_holiday]'))

    # add some more with a bit of preprocessing
    features.append('school_holiday')
    data['school_holiday'] = data['school_holiday'].astype(float)

    features.append('state_holiday')
    data.loc[data['state_holiday'] == 'a', 'state_holiday'] = '1'
    data.loc[data['state_holiday'] == 'b', 'state_holiday'] = '2'
    data.loc[data['state_holiday'] == 'c', 'state_holiday'] = '3'
    data['state_holiday'] = data['state_holiday'].astype(float)

    features.append('DayOfWeek')
    features.append('month')
    features.append('day')
    features.append('year')
    data['year'] = data.Date.apply(lambda x: x.split('-')[0])
    data['year'] = data['year'].astype(float)
    data['month'] = data.Date.apply(lambda x: x.split('-')[1])
    data['month'] = data['month'].astype(float)
    data['day'] = data.Date.apply(lambda x: x.split('-')[2])
    data['day'] = data['day'].astype(float)

    features.append('store_type')
    data.loc[data['store_type'] == 'a', 'store_type'] = '1'
    data.loc[data['store_type'] == 'b', 'store_type'] = '2'
    data.loc[data['store_type'] == 'c', 'store_type'] = '3'
    data.loc[data['store_type'] == 'd', 'store_type'] = '4'
    data['store_type'] = data['store_type'].astype(float)

    features.append('assortment')
    data.loc[data['assortment'] == 'a', 'assortment'] = '1'
    data.loc[data['assortment'] == 'b', 'assortment'] = '2'
    data.loc[data['assortment'] == 'c', 'assortment'] = '3'
    data['assortment'] = data['assortment'].astype(float)


print("Load the training, test and store data using pandas")
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
store = pd.read_csv("../input/store.csv")

print("Assume store open, if not provided")
test.fillna(1, inplace=True)

print("Consider only open stores for training. Closed stores wont count into the score.")
train = train[train["Open"] != 0]

print("Join with store")
train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')

features = []

print("augment features")
build_features(features, train)
build_features([], test)
print(features)

params = {"objective": "reg:linear",
          "eta": 0.2,
          "max_depth": 8,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "silent": 1
          }
num_trees = 900

print("Train a XGBoost model")
X_train, X_test = cross_validation.train_test_split(train, test_size=0.0125)
dtrain = xgb.DMatrix(X_train[features], np.log(X_train["Sales"] + 1))
dvalid = xgb.DMatrix(X_test[features], np.log(X_test["Sales"] + 1))
dtest = xgb.DMatrix(test[features])
watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
gbm = xgb.train(params, dtrain, num_trees, evals=watchlist, early_stopping_rounds=50, feval=rmspe_xg, verbose_eval=True)

print("Validating")
train_probs = gbm.predict(xgb.DMatrix(X_test[features]))
indices = train_probs < 0
train_probs[indices] = 0
error = rmspe(np.exp(train_probs) - 1, X_test['Sales'].values)
print('error', error)

print("Make predictions on the test set")
test_probs = gbm.predict(xgb.DMatrix(test[features]))
indices = test_probs < 0
test_probs[indices] = 0
submission = pd.DataFrame({"Id": test["Id"], "Sales": np.exp(test_probs) - 1})
submission.to_csv("xgboost_kscript_submission.csv", index=False)