try:
    from src.forecaster.abstract_forecaster import *
except ModuleNotFoundError:
    from .abstract_forecaster import *
import uuid
import pdb
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
EPS = 1


def early_stopping(train_step, val_losses, epochs):
    return train_step < 2000


def early_stopping_(train_step, val_losses, epochs):
    return len(val_losses) - np.argmax(val_losses) < EPOCHS_BEFORE_STOP


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
    def __init__(self, features_count=27, sess=None, plot_dir=None, batch_size=100):
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

    @abstractmethod
    def _build(self):
        # the name of all attributes defined here should stay the same in all child classes
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.features_count])
        self.true_sales = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        # forwarding
        self.output = None  # override

        # training
        # loss is Root Mean Square Percentage Error (RMSPE) (kaggle.com/c/rossmann-store-sales#evaluation)
        self.loss = tf.sqrt(tf.reduce_mean(tf.square((self.output - self.true_sales + EPS) / (self.true_sales + EPS))))
        optimizer = tf.train.AdamOptimizer()
        self.train_step = optimizer.minimize(self.loss)
        self.saver = tf.train.Saver([optimizer.variables()])

    def _predict_zero_if_closed(self):
        self.output *= self.input * tf.constant(
            np.eye(self.input.shape[1])[OPEN][None, ...]  # e_18 in shape (1 <broadcasts to #samples>, features_count)
            , dtype=tf.float32)

    def score(self, X, y):
        return self.sess.run(self.loss,
                             feed_dict={self.input: X, self.true_sales: y})

    def _decision_function(self, X):
        return self.sess.run(self.output, feed_dict={self.input: X})

    def _train(self, data):
        os.makedirs(".temp", exist_ok=True)
        name = self.__class__.__name__ + str(uuid.uuid4())[:5]

        print("({}) Start training".format(self.__class__.__name__))
        #  linear regression can't overfit, so as stopping criterion we take that the changes are small

        train_losses, val_losses, val_times = [np.inf], [np.inf], [np.inf]
        train_step = 0

        while early_stopping(train_step, val_losses, data.epochs): # no improvement in the last 10 epochs
            X, y = data.next_train_batch()
            train_loss, _ = self.sess.run([self.loss, self.train_step],
                                    feed_dict={self.input: X, self.true_sales: y})
            #  logging.info("({}) Step {}: Train loss {}".format(self.__class__.__name__, train_step, train_loss))
            print("({}) Step {}: Train loss {}".format(self.__class__.__name__, train_step, train_loss))
            train_losses.append(train_loss)

            if data.is_new_epoch or train_step % 100 == 0:
                val_loss = self.sess.run(self.loss,
                                         feed_dict={self.input: data.X_val, self.true_sales: data.y_val})
                val_losses.append(val_loss)

                if val_loss == min(val_losses[1:]):
                    self.saver.save(self.sess, ".temp/{}_params".format(name))
                    print("saved")

                val_times.append(train_step)
                print("({}) Step {}: Val loss {}".format(self.__class__.__name__, train_step, val_loss))
            train_step += 1

        self.saver.restore(self.sess, ".temp/{}_params".format(name))

        if self.plot_dir is not None:
            self._train_plot(train_losses, val_losses, val_times)

    def _train_plot(self, train_losses, val_losses, val_times):
        pdb.set_trace()
        plt.plot(range(len(train_losses)), train_losses, label="Train loss")
        plt.plot(val_times, val_losses, "o", label="Val loss")
        plt.savefig(self.plot_dir+"/training.png")
        plt.clf()

    def save(self):
        model_path = super().save()
        self.saver.save(self.sess, model_path+"_params")

    def load_params(self, file_name):
        self.saver.restore(self.sess, file_name)
        self.trained = True

    @staticmethod
    def load_model(file_name):
        model = AbstractForecaster.load_model(file_name)
        model.load_params(file_name+"_params")
        return model

