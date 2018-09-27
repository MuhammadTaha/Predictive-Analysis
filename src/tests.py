import unittest
import os

try:
    from src.forecaster.XGBForecaster import XGBForecaster
    from src.forecaster.naive_forecaster import NaiveForecaster, NaiveForecaster2
    from src.data.feedforward_data import Data
    from src.visualize_predictions import visualize_predictions
except ModuleNotFoundError:
    print("Use relative import without src")
    from forecaster.XGBForecaster import XGBForecaster
    from forecaster.naive_forecaster import NaiveForecaster, NaiveForecaster2
    from data import *
    from visualize_predictions import visualize_predictions_quick

import tensorflow as tf

src_dir = os.path.dirname(os.path.abspath(__file__))


class TestTraining(unittest.TestCase):
    # Tests to create a forecaster of the here listed classes and train it on our default Data class in toy mode
    Models = [NaiveForecaster, XGBForecaster]


    def train_forecaster(self, forecaster, data):
        """
        Trains the forecaster on the data
        :param data: a Data object
        :param forecaster: a forecaster object
        """
        forecaster.fit(data)
        print("Fitted {} on toy data".format(forecaster.__class__.__name__))
        visualize_predictions_quick(forecaster, src_dir + "/../plots/tests/{}".format(forecaster.__class__.__name__), store_id=1)

    def test_all_forecaster(self):
        # (1) creates a data object that will be used by all models
        # (2) iterarte over Models, create an instance and if necessary create a session for that instance

        # (1)
        data = Data(toy=True)

        # (2)
        for Model in TestTraining.Models:
            print("Test {}".format(Model.__name__))
            forecaster = Model()
            print("Successfully created {}".format(Model.__name__))

            if hasattr(forecaster, "sess"):
                sess = tf.Session()
                forecaster.sess = sess
                sess.run(tf.global_variables_initializer())

            self.train_forecaster(forecaster, data)

            if hasattr(forecaster, "sess"):
                sess.close()


class TestLoadTrainedModel(unittest.TestCase):
    """
    If you trained a model, please add a method that loads it here
    """

    def test_xgbmodel(self):
        print("Test trained xgb model")
        data = Data()
        forecaster = XGBForecaster.load_model("XGBForecaster2018-07-03-16:01")
        print("Loaded model.")
        print("Score of XGBForecaster2018-07-03-16:01: ", forecaster.score(data))




class TestData(unittest.TestCase):
    def test_random_batches(self):
        data = Data()

        samples = 0
        while data.epochs <3: # :)
            X, y = data.next_train_batch(100)
            self.assertEqual(X.shape[0], y.shape[0])
            self.assertEqual(X.shape[1], data.features_count)
            samples += X.shape[0]

