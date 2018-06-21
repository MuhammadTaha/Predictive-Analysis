from abc import ABC, abstractmethod
from sklearn.utils.validation import check_X_y, check_array
from sklearn.externals import joblib
import os
import datetime
from src.evaluate import Backtesting
from sklearn.model_selection import cross_val_score

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models")


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

    def fit(self, X, y=None):
        """
            Function to fit new models.
        :return: trained model
        """
        X, y = check_X_y(X, y, y_numeric=True, warn_on_dtype=True)
        self.trained = True
        return self._train(X, y)

    @abstractmethod
    def _train(self, X, y=None):
        """
            Override with your model training function
        """
        pass

    def predict(self, X):
        """
            Predict y values for input matrix X
        """
        assert self.trained, "Model is not trained cannot predict"
        X = check_array(X)
        return self._decision_function(self, X)

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

    def evaluate(self, score_function):
        """
            Backtesting is used to evaluate the total performance of the model after training
        :param score_function: function used to compare predictions and actualls 
        :return: float number from 0-100 representing the percentile score of the current model.
        """
        backtesting_instance = Backtesting(self, score_function)
        return backtesting_instance.evaluate()

    @abstractmethod
    def score(self, X, y=None):
        """
            used to get an impression of the performance of the current model.
        """
        return cross_val_score(self, X, y)

    def save(self):
        file_name = self.__class__.__name__ + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
        joblib.dump(self, os.path.join(MODEL_DIR, file_name))

    @staticmethod
    def load_model(file_name):
        return joblib.load(os.path.join(MODEL_DIR, file_name))
