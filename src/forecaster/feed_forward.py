from keras import Sequential
from keras.engine.saving import load_model
from keras.layers import Activation, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    from src.forecaster.abstract_forecaster import *
except ModuleNotFoundError:
    from .abstract_forecaster import *
import pdb
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


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

    def __init__(self, features_count=14, sess=None, plot_dir=None, batch_size=100):
        """
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

    def _predict_zero_if_closed(self):
        self.output *= self.input * tf.constant(
            np.eye(self.input.shape[1])[None, ...]  # e_18 in shape (1 <broadcasts to #samples>, features_count)
            , dtype=tf.float32)

    def score(self, X, y):
        return mean_absolute_error(y, self._decision_function(X))

    def _decision_function(self, X):
        return self.model.predict(X)

    def _train(self, data):
        X, y = data.all_train_data()
        history = self.model.fit(X, y, validation_split=0.2, epochs=10)

        # if self.plot_dir is not None:
        #     self._train_plot(train_losses, val_losses, val_times)

    def _train_plot(self, train_losses, val_losses, val_times):
        pdb.set_trace()
        plt.plot(range(len(train_losses)), train_losses, label="Train loss")
        plt.plot(val_times, val_losses, "o", label="Val loss")
        plt.savefig(self.plot_dir + "/training.png")
        plt.clf()

    def save(self):
        model_path = super().save()
        # self.saver.save(self.sess, model_path + "_params")
        self.model.save(model_path)

    def load_params(self, file_name):
        self.saver.restore(self.sess, file_name)
        self.trained = True

    @staticmethod
    def load_model(file_name):
        return load_model(file_name)

    def _build(self):
        self.model = Sequential()
        self.model.add(Dense(output_dim=150, input_shape=(self.features_count,)))
        self.model.add(Dropout(0.2))
        self.model.add(Activation('tanh'))
        self.model.add(Dense(output_dim=1))
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'acc'])
