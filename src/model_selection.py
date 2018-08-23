from data import Data
import numpy as np
import json
import os
import random
try:
    from .forecaster import *
except:
    from forecaster import *
import pdb

MODELS = [NaiveForecaster, XGBForecaster, LinearRegressor, FeedForwardNN1]
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../model_selection_results")

os.makedirs(RESULT_DIR, exist_ok=True)

data = Data()
print("got some data")

def estimate_score(model):
    # use the first 1000 batches only
    batches = np.random.permutation(list(range(100)))
    data.train_test_split(set(batches[:70]), set(batches[70:]))

    if hasattr(model, "sess"):
        sess = tf.Session()
        model.sess = sess
        sess.run(tf.global_variables_initializer())

    model.fit(data)

    if hasattr(model, "sess"):
        sess.close()

    score = []
    for batch_id in data.test_batch_ids:
        score.append(model.score(data.batches_X[batch_id], data.batches_y[batch_id]))
    return np.mean(score)


def best_hyperparams_and_score(model_class, num_combinations=2):
    """
    Creates a <model_class.__name__>-hyperparameters.json :
    [ {"params": <params>, "score": <score>}, ...]
    """
    params_choices  = model_class.params_grid

    # try different combinations
    # this is a random search in the paramater space, inspired by
    # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    # I don't use that one because I'm not sure if it can be used with our batch trained models
    def random_choice(key):
        if isinstance(params_choices[key], list):
            return random.choice(params_choices[key])
        assert isinstance(params_choices[key], np.ndarray), "{} has for param of unknown type for {}".format(model_class.__name__, key)
        return np.random.choice(params_choices[key])

    result, best_params, best_score, tried = [], {}, np.inf, []
    all_combinations = max(1, int(np.prod([len(params_choices[key]) for key in params_choices.keys()])))
    for i in range(min(num_combinations, all_combinations)):
        params = {key: random_choice(key) for key in params_choices.keys()}
        while params in tried:
            params = {key: random_choice(key) for key in params_choices.keys()}
        print("{} : try {}".format(model_class.__name__, params))
        model = model_class(**params)
        score = estimate_score(model)
        result.append({"params": params, "score": score})
        if score <= best_score:
            best_params, best_score = params, score

    result_path = os.path.join(RESULT_DIR, "{}-hyperparameters.json".format(model_class.__name__))
    #  save the results
    with open(result_path, "w") as f:
        result = json.dumps(result)
        print("result to be dumped", result)
        f.write(result)

    return best_params, best_score


def model_selection(extend_existing_results=True):
    """
    Find the best model, their hyperparameters and estiamted score
    generate a model_score.json that looks like this:
    { <class_name>: {"hyper_parameters": <hyperparameters>, "score": <score>}, ...}
    :return the best untrained model:
    """
    result_path = os.path.join(RESULT_DIR, 'model_selection.json')
    try: # load old results and don't run modelselection for them again
        assert extend_existing_results
        with open(result_path, "r") as f:
            result = json.load(f)
        print("Extend the following existing results of previous run: ", result)
    except (FileNotFoundError, AssertionError) as e:
        print("Full model selection because: ", type(e), e)
        result = {}  # here the json results will be collected

    def save_results(_result):
        with open(result_path, "w") as f:
            json_result = json.dumps(_result)
            print("result to be dumped", json_result)
            f.write(json_result)

    for model_class in MODELS:
        if model_class.__name__ in result.keys(): continue
        params, score = best_hyperparams_and_score(model_class)
        result[model_class.__name__] = {"hyper_parameters": params, "score": score}
        save_results(result)
        #
        # If we have an ExpertGroup model that trains one type of model per store
        # we only need to create the expert group for the model with the best
        # hyper parameters, so this is a good place to do it. We also need a way to
        # save the expert group into results and make it json dumpable and readable
        #

    # initialize the best model
    best_score = max([class_result["score"] for class_result in result.values()])
    best_class, best_params = [(class_name, result[class_name] )
                                for class_name in result.keys()
                                if result[class_name]["score"] == best_score][0]
    model_class = [model_class for model_class in MODELS if model_class.__name__==best_class][0]
    model = model_class(**best_params)
    return model


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
    train_best_model()