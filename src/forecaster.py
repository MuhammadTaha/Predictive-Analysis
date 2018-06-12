from abc import ABC, abstractmethod
from sklearn.utils.validation import check_X_y, check_array
from sklearn.externals import joblib
from sklearn import model_selection
import os
import datetime
from evaluate import Backtesting
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import logging

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models")

logger = logging.getLogger("forecaster")
logger.setLevel(logging.DEBUG)

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
        X, y = check_X_y(data.X_val, data.y_val, y_numeric=True, warn_on_dtype=True)
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
        # joblib.dump(self, os.path.join(MODEL_DIR, file_name))
        return os.path.join(MODEL_DIR, file_name)

    @staticmethod
    def load_model(file_name):
        return joblib.load(os.path.join(MODEL_DIR, file_name))


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape): return tf.Variable(tf.constant(0.1, shape=shape))


class FeedForward(AbstractForecaster):
    """
        Abstract class for feed forward models
        This class specifies:
        - how to handle the session
        - adds save/restore logic for tf.session
        - placeholders, loss and train_step should always have this name

        The session must be initialized from outside.
        If we do it from here, we would possibly reinitialize variables that were not defined here,
        eg we would destroy another trained model. (I couldn't find a way to initialize only the variables
        used by this class)
    """
    def __init__(self, features_count=25, sess=None, plot_dir=None, batch_size=100):
        """
            simple linear regression
        :param features_count: #features of X
        :param sess: tf.Session to use, per default, a new session will be created.
                    The session must be closed from outside
        :param plot_dir: dir for plots (eg of loss during training)
        :param batch_size: for the train batches
        """
        super().__init__()
        self.plot_dir = plot_dir
        if plot_dir is not None:
            os.makedirs(plot_dir, exist_ok=True)
        self.batch_size = batch_size
        self.features_count = features_count

        self.sess = sess if sess is not None else tf.Session()

        self._build()

    @abstractmethod
    def _build(self):
        # the name of all attributes defined here should stay the same in all child classes
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.features_count])
        self.true_sales = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        # forwarding
        self.output = None  # override

        # training
        # loss is Root Mean Square Percentage Error (RMSPE) (kaggle.com/c/rossmann-store-sales#evaluation)
        self.loss = tf.sqrt(tf.reduce_mean(tf.square((self.output - self.true_sales) / self.true_sales)))
        optimizer = tf.train.AdamOptimizer()
        self.train_step = optimizer.minimize(self.loss)
        self.saver = tf.train.Saver([optimizer.variables()])

    def score(self, X, y):
        return self.sess.run(self.loss,
                             feed_dict={self.input: X, self.true_sales: y})

    def _decision_function(self, X):
        return self.sess.run(self.output, feed_dict={self.input: X})

    def _train(self, data):
        # linear regression can't overfit, so as stopping criterion we take that the changes are small
        w_old, w_curr, b_old, b_curr = 0, 0, 0, 0
        train_losses, val_losses, val_times = [], [], []
        train_step = 0

        while np.allclose(w_old, w_curr) and np.allclose(b_old, b_curr) or data.epochs < 1:
            w_old, b_old = self.sess.run([self.weights, self.bias])
            X, y = data.next_train_batch(self.batch_size)
            train_loss, _ = self.sess.run([self.loss, self.train_step],
                                    feed_dict={self.input: X, self.true_sales: y})
            logging.info("({}) Step {}: Train loss {}".format(self.__class__.__name__, train_step, train_loss))
            train_losses.append(train_loss)
            if self.plot_dir is not None and data.is_new_epoch:
                val_loss = self.sess.run(self.loss,
                                         feed_dict={self.input: data.X_val, self.true_sales: data.y_val})
                val_losses.append(val_loss)
                val_times.append(train_step)
            w_curr, b_curr = self.sess.run([self.weights, self.bias])
            train_step += 1

        if self.plot_dir is not None:
            self._train_plot(train_losses, val_losses, val_times)

    def _train_plot(self, train_losses, val_losses, val_times):
        plt.plot(range(len(train_losses)), train_losses, label="Train loss")
        plt.plot(val_times, val_losses, "o", label="Val loss")
        plt.savefig(self.plot_dir+"/training.png")
        plt.clf()

    def save(self):
        model_path = super().save()
        self.saver.save(self.sess, model_path+"_params")

    def load_params(self, file_name):
        self.saver.restore(self.sess, file_name)

    @staticmethod
    def load_model(file_name):
        model = AbstractForecaster.load_model(file_name)
        model.load_params(file_name+"_params")
        return model


class LinearRegressor(FeedForward):
    def _build(self):
        # placeholders
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.features_count])
        self.true_sales = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        # forwarding
        self.weights = weight_variable([self.features_count, 1])
        self.bias = bias_variable([1, 1])
        self.output = tf.matmul(self.input, self.weights) + self.bias

        # training
        # loss is Root Mean Square Percentage Error (RMSPE) (kaggle.com/c/rossmann-store-sales#evaluation)
        self.loss = tf.sqrt(tf.reduce_mean(tf.square((self.output-self.true_sales)/self.true_sales)))
        optimizer = tf.train.AdamOptimizer()
        self.train_step = optimizer.minimize(self.loss)

        self.saver = tf.train.Saver([self.weights, self.bias])


class LinearLogRegressor(FeedForward):
    """
    Linear regression of log(y)
    """
    def _build(self):
        # placeholders
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.features_count])
        self.true_sales = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        # forwarding
        self.weights = weight_variable([self.features_count, 1])
        self.bias_1 = bias_variable([1, 1])
        self.bias_2 = bias_variable([1,1])
        self.output = tf.exp(tf.matmul(self.input, self.weights) + self.bias_1) + self.bias_2

        # training
        # loss is Root Mean Square Percentage Error (RMSPE) (kaggle.com/c/rossmann-store-sales#evaluation)
        self.loss = tf.sqrt(tf.reduce_mean(tf.square((self.output-self.true_sales)/self.true_sales)))
        optimizer = tf.train.AdamOptimizer()
        self.train_step = optimizer.minimize(self.loss)

        self.saver = tf.train.Saver([self.weights, self.bias])


class FeedForwardNN1(FeedForward):
    def __init__(self, features_count=25, sess=None, plot_dir=None, batch_size=100, hidden_features=[100], predict_logs=False):
        """
        :param hidden_features: list of how many neurons each hidden layer should have
        :param predict_logs: predict log(y)
        """
        super().__init(features_count=features_count, sess=sess, plot_dir=plot_dir, batch_size=batch_size)
        self.hidden_features = hidden_features
        self.predict_logs = predict_logs

    def _build(self):
        # placeholders
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.features_count])
        self.true_sales = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        """
        # forwarding
        self.weights_1 = weight_variable([self.features_count, 1])
        self.bias_1 = bias_variable([1, 1])
        self.out_1 = tf.nn.relu(tf.matmul(self.input, self.weights_1) + self.bias_1)

        self.weights_2 = weight_variable([self.features_count, 1])
        self.bias_2 = bias_variable([1, 1])
        self.output = tf.matmul(self.input, self.weights_1) + self.bias_1
        """
        # forwarding
        self.weights, self.biases = [], []
        features_in = self.features_count
        for neurons in self.hidden_features:
            self.weights.append(
                weight_variable([features_in, neurons])
            )
            self.biases.append(
                bias_variable([1, neurons])
            )
            features_in = neurons

        self.layers = []
        prev = self.input
        for w, b in zip(self.weights, self.biases):
            self.layers.append(
                tf.nn.relu(
                    tf.matmul(prev, w) + b
                )
            )
        self.weights.append(
            weight_variable([self.hidden_features[-1], 1])
        )
        self.biases.append(
            bias_variable([1,1])
        )

        if not self.predict_logs:
            self.output = tf.matmul(self.layers[-1], self.weights[-1]) + self.biases[-1]
        else:
            self.output = tf.exp(tf.matmul(self.layers[-1], self.weights[-1]) + self.biases[-1])

        # training
        # loss is Root Mean Square Percentage Error (RMSPE) (kaggle.com/c/rossmann-store-sales#evaluation)
        self.loss = tf.sqrt(tf.reduce_mean(tf.square((self.output-self.true_sales)/self.true_sales)))
        optimizer = tf.train.AdamOptimizer()
        self.train_step = optimizer.minimize(self.loss)

        self.saver = tf.train.Saver([self.weights, self.bias_1, self.bias_2])

