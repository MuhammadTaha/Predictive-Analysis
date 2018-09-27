import tensorflow as tf

try:
    from src.forecaster.feed_forward import *
except:
    from .feed_forward import *


class FeedForwardNN1(FeedForward):
    params_grid = {
        'hidden_features': [[100], [100, 100]],
        'predict_logs': [False] # [True, False] True gives nans all the time
    }

    def __init__(self, features_count=27, sess=None, plot_dir=None, batch_size=100, hidden_features=[100], predict_logs=False):
        """
        :param hidden_features: list of how many neurons each hidden layer should have
        :param predict_logs: predict log(y)
        """
        self.hidden_features = hidden_features
        self.predict_logs = predict_logs
        super().__init__(features_count=features_count, sess=sess, plot_dir=plot_dir, batch_size=batch_size)

    def _build(self):
        # placeholders
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.features_count])
        self.true_sales = tf.placeholder(dtype=tf.float32, shape=[None, 1])

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
            prev = self.layers[-1]

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

        self._predict_zero_if_closed()

        # training
        # loss is Root Mean Square Percentage Error (RMSPE) (kaggle.com/c/rossmann-store-sales#evaluation)
        self.loss = tf_rmspe(self.true_sales, self.output)
        optimizer = tf.train.AdamOptimizer()
        self.train_step = optimizer.minimize(self.loss)

        self.saver = tf.train.Saver(self.weights + self.biases)