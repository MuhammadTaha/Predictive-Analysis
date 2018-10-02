import keras
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from keras import Sequential, backend as K
from keras.engine.topology import Layer
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error

from src.forecaster.feed_forward import FeedForward

tfd = tfp.distributions


def elu_plus_one_plus_epsilon(x):
    """ELU activation with a very small addition to help prevent NaN in loss."""
    return (K.elu(x) + 1 + 1e-8)


class MDN(Layer):
    """A Mixture Density Network Layer for Keras.
    This layer has a few tricks to avoid NaNs in the loss function when training:
        - Activation for variances is ELU + 1 + 1e-8 (to avoid very small values)
        - Mixture weights (pi) are trained in as logits, not in the softmax space.
    A loss function needs to be constructed with the same output dimension and number of mixtures.
    A sampling function is also provided to sample from distribution parametrised by the MDN outputs.
    """

    def __init__(self, output_dimension, num_mixtures, **kwargs):
        self.output_dim = output_dimension
        self.num_mix = num_mixtures
        with tf.name_scope('MDN'):
            self.mdn_mus = Dense(self.num_mix * self.output_dim, name='mdn_mus')  # mix*output vals, no activation
            self.mdn_sigmas = Dense(self.num_mix * self.output_dim, activation=K.exp,
                                    name='mdn_sigmas')  # mix*output vals exp activation
            self.mdn_pi = Dense(self.num_mix, name='mdn_pi')  # mix vals, logits
        super(MDN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.mdn_mus.build(input_shape)
        self.mdn_sigmas.build(input_shape)
        self.mdn_pi.build(input_shape)
        self.trainable_weights = self.mdn_mus.trainable_weights + self.mdn_sigmas.trainable_weights + self.mdn_pi.trainable_weights
        self.non_trainable_weights = self.mdn_mus.non_trainable_weights + self.mdn_sigmas.non_trainable_weights + self.mdn_pi.non_trainable_weights
        super(MDN, self).build(input_shape)

    def call(self, x, mask=None):
        with tf.name_scope('MDN'):
            mdn_out = keras.layers.concatenate([self.mdn_mus(x),
                                                self.mdn_sigmas(x),
                                                self.mdn_pi(x)],
                                               name='mdn_outputs')
        return mdn_out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {
            "output_dimension": self.output_dim,
            "num_mixtures": self.num_mix
        }
        base_config = super(MDN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def split_mixture_params(params, output_dim, num_mixes):
    """Splits up an array of mixture parameters into mus, sigmas, and pis
    depending on the number of mixtures and output dimension."""
    mus = params[:num_mixes * output_dim]
    sigs = params[num_mixes * output_dim:2 * num_mixes * output_dim]
    pi_logits = params[-num_mixes:]
    return mus, sigs, pi_logits


def softmax(w, t=1.0):
    """Softmax function for a list or numpy array of logits. Also adjusts temperature."""
    e = np.array(w) / t  # adjust temperature
    e -= e.max()  # subtract max to protect from exploding exp values.
    e = np.exp(e)
    dist = e / np.sum(e)
    return dist


def get_mixture_loss_func(output_dim, num_mixes):
    """Construct a loss functions for the MDN layer parametrised by number of mixtures."""

    # Construct a loss function with the right number of mixtures and outputs
    def loss_func(y_true, y_pred):
        out_mu, out_sigma, out_pi = tf.split(y_pred, num_or_size_splits=[num_mixes * output_dim,
                                                                         num_mixes * output_dim,
                                                                         num_mixes],
                                             axis=1, name='mdn_coef_split')
        cat = tfd.Categorical(logits=out_pi)
        component_splits = [output_dim] * num_mixes
        mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)
        sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)
        coll = [tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
                in zip(mus, sigs)]
        mixture = tfd.Mixture(cat=cat, components=coll)
        loss = mixture.log_prob(y_true)
        loss = tf.negative(loss)
        loss = tf.reduce_mean(loss)
        # print("loss: {}, sigma: {}".format(loss, sigs))
        return loss

    # Actually return the loss_func
    with tf.name_scope('MDN'):
        return loss_func


def sample_from_categorical(dist):
    """Samples from a categorical model PDF."""
    r = np.random.rand(1)  # uniform random number in [0,1]
    accumulate = 0
    for i in range(0, dist.size):
        accumulate += dist[i]
        if accumulate >= r:
            return i
    tf.logging.info('Error sampling mixture model.')
    return -1


def sample_from_output(params, output_dim, num_mixes, temp=1.0):
    """Sample from an MDN output with temperature adjustment."""
    mus = params[:num_mixes * output_dim]
    sigs = params[num_mixes * output_dim:2 * num_mixes * output_dim]
    pis = softmax(params[-num_mixes:], t=temp)
    m = sample_from_categorical(pis)
    # Alternative way to sample from categorical:
    # m = np.random.choice(range(len(pis)), p=pis)
    mus_vector = mus[m * output_dim:(m + 1) * output_dim]
    sig_vector = sigs[m * output_dim:(m + 1) * output_dim] * temp  # adjust for temperature
    # print(sig_vector)
    # print("mu: {}\nsigma: {}\npis: {}\n==================\n".format(mus, sigs, pis))
    # print("=====================================")
    cov_matrix = np.identity(output_dim) * sig_vector
    sample = np.random.multivariate_normal(mus_vector, cov_matrix, 1)
    return sample


def output_layer(args):
    pass


class MDNetwork(FeedForward):
    MDN_COMPONENTS = 5

    def _build(self):
        """
            Builds a 3 stages forecaster based on paper Associative and Recurrent Mixture density networks
            The network expects an input of shape (batch_size, n_past_weeks, n_features).
            First component: is an associative layer with 3 hidden layers. first is an input layer that is a merge of
                            numerical features layer and categorical features layer. second is a dense layer with 50 units.
                            lastly is a dropout layer with .5 drop_rate
            Second component: is a recurrent layer built with two lstms with .5 drop  rate then a reshape that changes
                            the output of lstm from (batch_size, n_past_weeks, n_features) to
                            (batch_size, n_past_weeks * n_features)
            third component: is a Mixture density layer that outputs a Gaussian distribution of sales based on past
                            sales history
        :return: keras model of output dimension (mdn_components * (mdn_output_dim + 2), )
        """
        # Network configs
        dropout_rate = .5

        learning_rate = 0.001
        n_hidden_units = 50

        learning_rate = 0.01
        MDN_OUTPUT_DIM = 1

        self.model = Sequential()
        self.model.add(Dense(output_dim=50, input_shape=(self.features_count,), activation='tanh'))
        self.model.add(Dense(output_dim=50, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(MDN(MDN_OUTPUT_DIM, self.MDN_COMPONENTS))
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'acc'])

        # model = Model(outputs=output_layer)
        self.model.compile(loss=get_mixture_loss_func(MDN_OUTPUT_DIM, self.MDN_COMPONENTS),
                           optimizer=Adam(learning_rate), metrics=['mae', 'acc'])

    def _decision_function(self, X):
        y_pred = self.model.predict(X)
        # mus = np.apply_along_axis((lambda a: a[:self.MDN_COMPONENTS]), 1, y_pred)
        # sigs = np.apply_along_axis((lambda a: a[self.MDN_COMPONENTS:2 * self.MDN_COMPONENTS]), 1, y_pred)
        # pis = np.apply_along_axis((lambda a: softmax(a[2 * self.MDN_COMPONENTS:])), 1, y_pred)
        y_pred = np.apply_along_axis(sample_from_output, 1, y_pred, 1, self.MDN_COMPONENTS, temp=1.0).reshape(
            X.shape[0])

        return y_pred

    def _train(self, data):
        X, y = data.all_train_data()
        history = self.model.fit(X, y, validation_split=0.2, epochs=50)

    def score(self, X, y):
        y_pred = self.model.predict(X)
        mus = np.apply_along_axis((lambda a: a[:self.MDN_COMPONENTS]), 1, y_pred)
        sigs = np.apply_along_axis((lambda a: a[self.MDN_COMPONENTS:2 * self.MDN_COMPONENTS]), 1, y_pred)
        pis = np.apply_along_axis((lambda a: softmax(a[2 * self.MDN_COMPONENTS:])), 1, y_pred)
        y_pred = np.apply_along_axis(sample_from_output, 1, y_pred, 1, self.MDN_COMPONENTS, temp=1.0).reshape(
            y.shape)
        s = np.sum(mus * pis, axis=1)
        print(mean_absolute_error(y, y_pred), mean_absolute_error(y, s))
        return mean_absolute_error(y, y_pred)
