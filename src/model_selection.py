from data import Data
import numpy as np
try:
    from .forecaster import *
except:
    from forecaster import *
import pdb

def estimate_score(model, data):
    # use the first 1000 batches only
    batches = np.random.permutation(list(range(1000)))
    data.train_test_split(set(batches[:700]), set(batches[700:]))
    pdb.set_trace()

    model.fit(data)

    score = []
    for batch_id in data.test_batch_ids:
        score.append(model.score(data.batch_X[batch_id], data.batch_y[batch_id]))
    return np.mean(score)

def test_estimate_score():
    model = NaiveForecaster()
    print("Score: ", estimate_score(model, Data(update_disk=True)))

def main():
    test_estimate_score()
