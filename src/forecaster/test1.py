from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
import random
import sys
sys.path.append('/home/mshaban/DeployedProjects/Predictive-Analysis')
from src.forecaster.mdn import MDNetwork
from src.forecaster import *

forecasters = [MDNetwork ]  # ,MDNetwork(), XGBForecaster(), FeedForward(),  SVRForecaster(gamma='auto', epsilon=0.1)]
train_path = "/home/mshaban/DeployedProjects/Predictive-Analysis/models/results/final_train.csv"
test_path = "/home/mshaban/DeployedProjects/Predictive-Analysis/models/results/final_test.csv"
feed_forward_data = FeedForwardData(train_path=train_path, test_path=test_path)

NUM_POINTS_FOR_ESTIMATE = int(feed_forward_data.data.shape[0] * 0.5)
# NUM_POINTS_FOR_ESTIMATE =100
points = list(range(NUM_POINTS_FOR_ESTIMATE))
# points = np.random.permutation(points)
split = int(0.95 * len(points))

small_points = list(range(int(0.2 * NUM_POINTS_FOR_ESTIMATE)))
small_split = int(0.95 * len(small_points))

feed_forward_data.train_test_split(set(points[:split]), set(points[split:]))
xt, yt = feed_forward_data.all_test_data()
# feed_forward_data.data.to_csv("/home/mshaban/DeployedProjects/Predictive-Analysis/models/results/final_train.csv", index=False)
# feed_forward_data.final_test.to_csv("/home/mshaban/DeployedProjects/Predictive-Analysis/models/results/final_test.csv", index=False)


model_score_df = pd.DataFrame.from_dict({'model_name': [], 'params': [], 'score': []})
params_validation_df = feed_forward_data.data.iloc[points][feed_forward_data.data.Store == 5]
trial = random.randint(1, 100)
HYPER_PARAM_EPOCHS = 100
HYPER_PARAM_ROUNDS = 100
FINAL_TRAINING_EPOCHS = 4000
FINAL_TRAINING_ROUNDS = 3000


def search_hyper_params(model_class):
    global model_score_df, feed_forward_data, params_validation_df, HYPER_PARAM_EPOCHS, HYPER_PARAM_ROUNDS
    feed_forward_data.train_test_split(
        set(small_points[:small_split]), set(small_points[small_split:]))
    params_choices = model_class.params_grid
    num_combinations = 10
    pred_param_df = pd.DataFrame.from_dict({"params": [], "pred": []})

    def random_choice(key):
        return np.random.choice(params_choices[key])

    result, best_params, best_score, tried = [], {}, np.inf, []
    all_combinations = max(1, int(np.prod([len(params_choices[key]) for key in params_choices.keys()])))
    for i in range(min(num_combinations, all_combinations)):
        params = {key: random_choice(key) for key in params_choices.keys()}
        while params in tried:
            params = {key: random_choice(key) for key in params_choices.keys()}
        model = model_class(**params)
        print("params: {}".format(params))
        history = model.fit(feed_forward_data, epochs=HYPER_PARAM_EPOCHS, n_rounds=HYPER_PARAM_ROUNDS)
        if history and 'loss' in history.history and np.any(np.isnan(history.history['loss'])):
            continue
        score = model.score(xt, yt)
        model_score_df = model_score_df.append({'params': params, 'score': score, 'model_name': model_class.__name__},
                                               ignore_index=True)
        print("params: {}, score: {}".format(params, score))
        if score <= best_score:
            best_params, best_score = params, score
        pred = model._decision_function(params_validation_df[FEATURES].values)
        pred_param_df = pred_param_df.append({'params': params, 'pred': pred}, ignore_index=True)

    feed_forward_data.train_test_split(set(points[:split]), set(points[split:]))
    pred_param_df.to_csv(
        "/home/mshaban/DeployedProjects/Predictive-Analysis/models/results/hyper-params-{}-{}.csv".format(
            model_class.__name__, trial), index=False)
    model_score_df.to_csv(
        "/home/mshaban/DeployedProjects/Predictive-Analysis/models/results/model_score_{}-{}.csv".format(
            forecasters[0].__name__, trial))
    model = model_class(**best_params)
    return model

def train_forecaster():
    def run_forecaster(forecaster):
        # if forecaster == forecasters[0]:
        #     pass
        forecaster = search_hyper_params(forecaster)
        forecaster.fit(feed_forward_data, epochs=FINAL_TRAINING_EPOCHS, n_rounds=FINAL_TRAINING_ROUNDS)
        forecaster.save()
        score = forecaster.score(xt, yt)
        print("{}, {}".format(type(forecaster).__name__, score))
        test = feed_forward_data.final_test
        test_probs = forecaster._decision_function(test[FEATURES].values)
        indices = test_probs < 0
        test_probs[indices] = 0

        submission = pd.DataFrame({"Id": test["Id"], "Sales": np.exp(test_probs) - 1})


        # closed stores sale none
        def fix_closed(row):
            if test[test['Id'] == row['Id']]['Open'].values[0] == 0:
                return 0
            else:
                return row['Sales']


        submission['Sales'] = submission.apply(fix_closed, axis=1)
        submission.to_csv("/home/mshaban/DeployedProjects/Predictive-Analysis/models/results/{}-{}.csv".format(
            type(forecaster).__name__, trial), index=False)
        print("{}, {}".format(type(forecaster).__name__, score))
        print(trial)

    for forecaster in forecasters:
        run_forecaster(forecaster)
    print("results")

def load_forecaster():
    forecaster = MDNetwork.load_model("/home/mshaban/DeployedProjects/Predictive-Analysis/models/MDNetwork2018-10-04-15-29")
    score = forecaster.score(xt, yt)
    print(score)


train_forecaster()