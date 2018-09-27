from abc import ABC, abstractmethod
from sklearn.utils.validation import check_X_y, check_array
from sklearn.externals import joblib
import os
import datetime
import tensorflow as tf
import numpy as np
import pdb

import logging
try:
    from src.data.feature_enum import *
except ModuleNotFoundError:
    from data.feature_enum import *


MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../models")

logger = logging.getLogger("forecaster")
logger.setLevel(logging.DEBUG)

EPOCHS_BEFORE_STOP = 10  # number of epochs with no improvement before training is stopped

EPS = 50

OPEN = 4

def rmspe(sales, prediction):
    return np.sqrt(np.mean(np.square(
        (prediction - sales + EPS) / (sales + EPS)
    )))


def tf_rmspe(sales, prediction):
    print("sales", sales.shape, "prediction", prediction.shape)
    return tf.sqrt(tf.reduce_mean(tf.square(
        (prediction - sales + EPS) / (sales + EPS)
    )))


def early_stopping_debug(train_step, val_losses, epochs):
    return train_step < 2000


def early_stopping(train_step, val_losses, epochs):
    return len(val_losses) - np.argmax(val_losses) < EPOCHS_BEFORE_STOP


class AbstractForecaster(ABC):
    """
        All forecaster should extend this class. steps to extend abstract forecaster 
        1. add __init__ function with your kwargs
        2. don't change fit function and override the _train function
        3. it goes as well for predict, add your prediction logic to _decision_function
        4. in case your approach will use NN pleace consider keras api and use the build function to build your model.
        5. the score function is used as a to test on small data sets and get an impression of the performance of your model
            to get proper testing result use the evaluate function
        6. save/load functions that can be used for these purposes.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """
            Model initialization
        """
        self.trained = False
        pass

    def fit(self, data):
        """
            Function to fit new models.
        :return: trained model
        """
        try:
            X, y = check_X_y(data.X_val, data.y_val, y_numeric=True, warn_on_dtype=True)
        except Exception as e:
            # this check fails for lstm shaped input, so let's print it but allow it
            print("check_X_y:", type(e), e)
        self.trained = True
        return self._train(data)

    @abstractmethod
    def _train(self, data):
        """
            Override with your model training function
        """
        pass

    def predict(self, X):
        """
            Predict y values for input matrix X
            y.shape = (#samples, 1)
        """
        assert self.trained, "Model is not trained cannot predict"
        #  X = check_array(X)
        y = self._decision_function(X)

        try:
            assert (y == y * X[:, OPEN]).all()
        except AssertionError:
            print("({}) Warning: Original prediction not zero for rows where stores are closed")
        except Exception as e:
            print(type(e), e); pdb.set_trace()

        if len(X.shape) == 2:  # feed forward data
            filter_open = X[:, OPEN]
        elif len(X.shape) == 3:  # lstm data
            filter_open = X[:, -1, OPEN]
        else:
            raise Warning("X shape is weird")
            pdb.set_trace()

        return y * filter_open


    @abstractmethod
    def _decision_function(self, X):
        """
            Here comes the logic of your predictions
        """
        pass

    @abstractmethod
    def _build(self):
        """
            This function is intended to only be used when building NN based models. it's recommended to use Keras api
            to build the model
        :return: keras model
        """
        pass

    """
    def evaluate(self, score_function):
        "" "
            Backtesting is used to evaluate the total performance of the model after training
        :param score_function: function used to compare predictions and actualls 
        :return: float number from 0-100 representing the percentile score of the current model.
        "" "
        backtesting_instance = Backtesting(self, score_function)
        return backtesting_instance.evaluate()
    """

    @abstractmethod
    def score(self, X, y):
        """
            used to get an impression of the performance of the current model.
        """
        p = self.predict(X)
        return rmspe(y, p)

    def save(self):
        file_name = self.__class__.__name__ + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
        joblib.dump(self, os.path.join(MODEL_DIR, file_name))
        return os.path.join(MODEL_DIR, file_name)

    @staticmethod
    def load_model(file_name):
        return joblib.load(os.path.join(MODEL_DIR, file_name))




