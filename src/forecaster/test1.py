import random

import numpy as np
import pandas as pd

from src.forecaster import FEATURES, FeedForwardData, SVRForecaster, EPOCHS
from src.forecaster.feed_forward import FeedForward
import pandas as pd

forecasters = [FeedForward, ]  # ,MDNetwork(), XGBForecaster(), FeedForward(),  SVRForecaster(gamma='auto', epsilon=0.1)]
feed_forward_data = FeedForwardData()
NUM_POINTS_FOR_ESTIMATE = feed_forward_data.data.shape[0]
# NUM_POINTS_FOR_ESTIMATE =100
points = list(range(NUM_POINTS_FOR_ESTIMATE))
# points = np.random.permutation(points)
split = int(0.95 * len(points))

small_points = list(range(int(0.2*NUM_POINTS_FOR_ESTIMATE)))
small_split = int(0.95 * len(small_points))

feed_forward_data.train_test_split(set(points[:split]), set(points[split:]))
xt, yt = feed_forward_data.all_test_data()
# feed_forward_data.data.to_csv("/home/mshaban/DeployedProjects/Predictive-Analysis/models/results/final_train.csv", index=False)
# feed_forward_data.final_test.to_csv("/home/mshaban/DeployedProjects/Predictive-Analysis/models/results/final_test.csv", index=False)


results = []
model_score_df = pd.DataFrame.from_dict({'model_name': [], 'params': [],'score': [] })

def search_hyper_params(model_class, data):
    global model_score_df
    feed_forward_data.train_test_split(
        set(small_points[:small_split]), set(small_points[small_split:]))
    params_choices = model_class.params_grid
    num_combinations = 1

    def random_choice(key):
        return np.random.choice(params_choices[key])

    result, best_params, best_score, tried = [], {}, np.inf, []
    all_combinations = max(1, int(np.prod([len(params_choices[key]) for key in params_choices.keys()])))
    for i in range(min(num_combinations, all_combinations)):
        params = {key: random_choice(key) for key in params_choices.keys()}
        while params in tried:
            params = {key: random_choice(key) for key in params_choices.keys()}
        model = model_class(**params)
        model.fit(data, epochs=2)
        score = model.score(xt, yt)
        model_score_df = model_score_df.append({'params': params, 'score': score, 'model_name': model_class.__name__}, ignore_index=True)
        print("params: {}, score: {}".format(params, score))
        if score <= best_score:
            best_params, best_score = params, score

    feed_forward_data.train_test_split(set(points[:split]), set(points[split:]))

    model = model_class(**best_params)
    return model

trial = random.randint(1, 10)

for forecaster in forecasters:
    # if forecaster == forecasters[0]:
    #     pass
    forecaster = search_hyper_params(forecaster, feed_forward_data)
    # else:
    forecaster.fit(feed_forward_data, epochs=1)

    forecaster.save()
    score = forecaster.score(xt, yt)
    results.append("{}, {}".format(type(forecaster).__name__, score))
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
    print(results)
model_score_df.to_csv("/home/mshaban/DeployedProjects/Predictive-Analysis/models/results/model_score_results-{}.csv".format(trial))