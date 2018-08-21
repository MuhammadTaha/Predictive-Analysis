from data import Data
import numpy as np
import json
import os
try:
    from .forecaster import *
except:
    from forecaster import *
import pdb

MODELS = [NaiveForecaster, XGBForecaster, LinearRegressor, FeedForwardNN1]
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../model_selection_results")

os.makedirs(RESULT_DIR, exist_ok=True)

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
    data = Data(update_disk=True)
    print("Got some data")
    print("Score: ", estimate_score(model, data))


def model_selection():
    """
    Find the best model, their hyperparameters and estiamted score
    generate a model_score.json that looks like this:
    { <class_name>: {"hyper_parameters": <hyperparameters>, "score": <score>}, ...}
    :return the best untrained model:
    """
    result_path = os.path.join(RESULT_DIR, 'model_selection.json')
    try: # load old results and don't run modelselection for them again
        with open(result_path, "r") as f:
            result = json.load(f)
    except FileNotFoundError:
        result = {} # here the json results will be collected

    for model_class in MODELS:
        if model_class in result.keys(): continue
        params, score = best_hyperparams_and_score(model_class)
        result[model_class.__name__] = {"hyper_parameters": params, "score": score}

    #  save the results
    with open(result_path, "w") as f:
        result = json.dumps(result)
        print("result to be dumped", result)     
        f.write(result)



def train_best_model():
    model = model_selection()
    data = Data()
    model.fit(data)
    print("Save the best trained model")
    try:
        path = model.save()
        print("Saved to ", path)
        with open("best_model.txt", "w") as file:
            file.write(path)
    except Exception as e:
        print("While saving the best model, an error has occurred")
        print(type(e))
        print(e)
        pdb.set_trace()


def main():
    test_estimate_score()
