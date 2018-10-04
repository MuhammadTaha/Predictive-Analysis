from keras import Sequential, callbacks
from keras.engine.saving import load_model
from keras.layers import BatchNormalization, Dense
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error

try:
    from src.forecaster.abstract_forecaster import *
except ModuleNotFoundError:
    from .abstract_forecaster import *
import tensorflow as tf
import numpy as np


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape): return tf.Variable(tf.constant(0.1, shape=shape))


class FeedForward(AbstractForecaster):
    params_grid = {
        "units": np.linspace(100, 200, num=5).astype(int),
        "activation": ["tanh", "sigmoid"],
        # "drop_out": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "n_layers": [1, 2, 3, 4, 5],
        "kernel_regularizer": ["l1_l2", "l1", "l2", None]
    }

    def __init__(self, features_count=FEATURE_COUNT, **kwargs):
        """
        :param features_count: #features of X
        :param sess: tf.Session to use, per default, a new session will be created.
                    The session must be closed from outside
        :param plot_dir: dir for plots (eg of loss during training)
        :param batch_size: for the train batches
        """
        super().__init__()
        self.features_count = features_count
        if not self.loaded:
            if 'n_layers' in kwargs:
                self.n_layer = kwargs.pop("n_layers")

            self.params = kwargs
            self._build()

    def score(self, X, y):
        return mean_absolute_error(y, self._decision_function(X))

    def _decision_function(self, X):
        return np.squeeze(self.model.predict(X))

    def _train(self, data, **kwargs):
        X, y = data.all_train_data()
        early_callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=0, mode='auto',
                                                 baseline=None, )
        history = self.model.fit(X, y, validation_split=0.2, epochs=kwargs["epochs"], callbacks=[early_callback])

        return history

    def save(self, trial):
        file_name = "{}-{}".format(self.__class__.__name__, trial)
        self.model.save(os.path.join(MODEL_DIR, file_name))

    @staticmethod
    def load_model(file_name):
        return load_model(file_name)

    def _build(self, ):
        self.model = Sequential()
        self.model.add(BatchNormalization(axis=1))
        for _ in range(self.n_layer):
            if _ == 0:
                self.model.add(
                    Dense(input_shape=(self.features_count,), **self.params, ))
            else:
                self.model.add(Dense(**self.params))
                # self.model.add(Dropout(self.drop_out))
        self.model.add(Dense(units=1, activation="relu"))
        self.model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae'])
