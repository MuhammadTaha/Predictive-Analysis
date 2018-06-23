import unittest
from data import Data

class TestData(unittest.TestCase):
    def test_random_batches(self):
        data = Data(p_train=0.1, p_val=0.45, p_test=0.45)

        samples = 0
        while data.epochs <3: # :)
            X, y = data.next_train_batch(100)
            self.assertEqual(X.shape[0], y.shape[0])
            self.assertEqual(X.shape[1], data.features_count)
            samples += X.shape[0]

        self.assertEqual(samples-len(data.used_this_epoch), 3*len(data.train_row_ids))
