from .feed_forward import *


class LinearRegressor(FeedForward):
    def _build(self):
        # placeholders
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.features_count])
        self.true_sales = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        # forwarding
        self.weights = weight_variable([self.features_count, 1])
        self.bias = bias_variable([1, 1])
        self.output = tf.matmul(self.input, self.weights) + self.bias
        self._predict_zero_if_closed()

        # training
        # loss is Root Mean Square Percentage Error (RMSPE) (kaggle.com/c/rossmann-store-sales#evaluation)
        self.loss = tf.sqrt(tf.reduce_mean(tf.square((self.output - self.true_sales + EPS) / (self.true_sales + EPS))))
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
        self.loss = tf.sqrt(tf.reduce_mean(tf.square((self.output-self.true_sales+EPS)/(self.true_sales+EPS))))
        optimizer = tf.train.AdamOptimizer()
        self.train_step = optimizer.minimize(self.loss)

        self.saver = tf.train.Saver([self.weights, self.bias])