try:
    from src.forecaster.abstract_forecaster import *
except ModuleNotFoundError:
    from .abstract_forecaster import *
from sklearn.svm import SVR
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow as tf

import pdb


class LSTMForecaster(AbstractForecaster):
    params_grid = {
        "hidden_units": list(range(30, 200, 10)),
        "dropout": [0, 0.2, 0.4, 0.6, 0.8],
        "recurrent_dropout": [0, 0.2, 0.4, 0.6, 0.8]
    }

    def __init__(self, num_timesteps, features_count, hidden_units=64, dropout=0, recurrent_dropout=0):
        model = Sequential()
        # model.add(Embedding(max_features, 128))
        model.add(LSTM(hidden_units,
                       activation='relu',
                       dropout=dropout,
                       recurrent_dropout=recurrent_dropout,
                       input_shape=(num_timesteps, features_count)
                       ))
        model.add(Dense(1, activation='linear'))

        model.compile(loss=tf_rmspe, optimizer='adam', metrics=['accuracy', 'mse'])

        self.model = model

    def _decision_function(self, X):
        return self.model.predict(X)

    def _train(self, data):
        X, y = data.all_train_data()
        print("Fit LSTM with X: {} and y: {}".format(np.array(X).shape, np.array(y).shape))
        self.model.fit(X, y, epochs=10, sample_weight=X[:, -1, OPEN], batch_size=32)

    def _build(self):
        pass

    def score(self, X, y):
        prediction = self.predict(X)
        return rmspe(y, prediction)

    def save(self):
        file_name = self.__class__.__name__ + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
        params = self.model.get_params()
        joblib.dump(params, os.path.join(MODEL_DIR, file_name))
        return os.path.join(MODEL_DIR, file_name)

    @staticmethod
    def load_model(file_name):
        lstm = LSTM()
        lstm.model.set_params(joblib.load(os.path.join(MODEL_DIR, file_name)))
        return lstm
