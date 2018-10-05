import sys

sys.path.append('/home/mshaban/DeployedProjects/Predictive-Analysis')
from src.data.data_extraction import DataExtraction

import ast
import pandas as pd
import random

from src.forecaster.mdn import MDNetwork
from src.forecaster import *

RESULTS_DIR = "/home/mshaban/DeployedProjects/Predictive-Analysis/models/results/"
TRAIN_DATA_CSV = RESULTS_DIR + "final_train.csv"
TEST_DATA_CSV = RESULTS_DIR + "final_test.csv"
TRIAL = random.randint(1, 100)
HYPER_PARAM_EPOCHS = 100
HYPER_PARAM_ROUNDS = 100
FINAL_TRAINING_EPOCHS = 500
FINAL_TRAINING_ROUNDS = 1000

DATA = DataExtraction(train_path=TRAIN_DATA_CSV, test_path=TEST_DATA_CSV)
FORECASTERS = [MDNetwork, XGBForecaster, FeedForward, SVRForecaster]
HYPER_PARAMS_SPLIT = int(DATA.data.shape[0] * 0.1)
# HYPER_PARAMS_SPLIT = 4000
N_COMBINATION = 10

X, Y = DATA.data[FEATURES].values, DATA.data.Sales.values

mini_split = int(1.25 * HYPER_PARAMS_SPLIT)
X_train, y_train = X[:HYPER_PARAMS_SPLIT], Y[:HYPER_PARAMS_SPLIT]
X_test, y_test = X[HYPER_PARAMS_SPLIT:mini_split], Y[HYPER_PARAMS_SPLIT:mini_split]

DATA.data['trained'] = False
DATA.data.iloc[:HYPER_PARAMS_SPLIT]['trained'] = True
params_validation_df = DATA.data[DATA.data.Store == 5]


def save_results_df(df, file_name):
    df.to_csv(RESULTS_DIR + file_name, index=False)



# save_results_df(DATA.data, "final_train.csv")
# save_results_df(DATA.final_test, "final_test.csv")

def search_hyper_params(model_class):
    global DATA, params_validation_df, HYPER_PARAM_EPOCHS, HYPER_PARAM_ROUNDS
    params_choices = model_class.params_grid

    def random_choice(key):
        return np.random.choice(params_choices[key])

    pred_param_df = pd.DataFrame.from_dict({"params": [], "pred": []})
    model_score_df = pd.DataFrame.from_dict({'model_name': [], 'params': [], 'score': []})

    result, best_params, best_score, tried = [], {}, np.inf, []
    all_combinations = max(1, int(np.prod([len(params_choices[key]) for key in params_choices.keys()])))
    for i in range(min(N_COMBINATION, all_combinations)):
        params = {key: random_choice(key) for key in params_choices.keys()}
        while params in tried:
            params = {key: random_choice(key) for key in params_choices.keys()}
        model = model_class(**params)
        print("params: {}".format(params))
        history = model.fit(X_train, y_train, epochs=HYPER_PARAM_EPOCHS, n_rounds=HYPER_PARAM_ROUNDS)
        if history and 'loss' in history.history and np.any(np.isnan(history.history['loss'])):
            continue
        score = model.score(X_test, y_test)
        model_score_df = model_score_df.append({'params': params, 'score': score, 'model_name': model_class.__name__},
                                               ignore_index=True)
        print("combination: {}/{}, params: {}, score: {}".format(i, N_COMBINATION, params, score))
        if score <= best_score:
            best_params, best_score = params, score
        pred = model._decision_function(params_validation_df[FEATURES].values)
        pred_param_df = pred_param_df.append({'params': params, 'pred': pred}, ignore_index=True)

    save_results_df(pred_param_df, "hyper-params-pred-{}-{}.csv".format(
        model_class.__name__, TRIAL))
    save_results_df(model_score_df, "model_score_{}-{}.csv".format(
        model_class.__name__, TRIAL))
    model = model_class(**best_params)
    return model, best_params


def run_forecaster(forecaster):
    # Run Hyperparams randomized search
    forecaster.fit(X, Y, epochs=FINAL_TRAINING_EPOCHS, n_rounds=FINAL_TRAINING_ROUNDS)
    forecaster.save(TRIAL)
    score = forecaster.score(X_test, y_test)

    # Run to get final kaggle predictions
    test = DATA.final_test
    test_probs = forecaster._decision_function(test[FEATURES].values)
    # indices = test_probs < 0
    # test_probs[indices] = 0
    submission = pd.DataFrame({"Id": test["Id"], "Sales": test_probs})
    submission['Sales'] = submission['Sales'] + test.AvgSales

    # closed stores sale none
    def fix_closed(row):
        test_row = test[test['Id'] == row['Id']]
        if test_row['Open'].values[0] == 0:
            return 0
        else:
            return row['Sales']

    submission['Sales'] = submission['Sales'].apply(lambda x: np.exp(x-1))
    submission['Sales'] = submission.apply(fix_closed, axis=1)


    save_results_df(submission, "{}-{}-final-pred.csv".format(type(forecaster).__name__, TRIAL))
    print("{}, {}".format(type(forecaster).__name__, score))
    print(TRIAL)
    return type(forecaster).__name__, score


def load_best_params(forecaster, trial):
    forecaster_param_scores = pd.DataFrame.from_csv(
        RESULTS_DIR + "model_score_{}-{}.csv".format(forecaster.__name__, trial))
    forecaster_param_scores = forecaster_param_scores.reset_index()
    forecaster_param_scores['params'] = forecaster_param_scores['params'].apply(lambda x: ast.literal_eval(x))
    min_score_idx = forecaster_param_scores['score'].idxmin()

    return forecaster_param_scores['params'].iloc[min_score_idx], forecaster_param_scores['score'].min()




if __name__ == '__main__':
    load_trial = False
    arg = sys.argv[1]
    if len(sys.argv) > 2:
        load_trial = True
        TRIAL = int(sys.argv[2])
    final_model_results = pd.DataFrame.from_dict({"model": [], "score": [], "params": []})
    for _, forecaster in enumerate(FORECASTERS):
        if not arg in forecaster.__name__:
            print(forecaster.__name__)
            continue
        if load_trial:
            params = load_best_params(forecaster, TRIAL)[0]
            forecaster = forecaster(**params)
            model, score = run_forecaster(forecaster)
        else:
            forecaster, params = search_hyper_params(forecaster)
            model, score = run_forecaster(forecaster)
        final_model_results = final_model_results.append({"model": model, "score": score, "params": params},
                                                         ignore_index=True)

    save_results_df(final_model_results, "best-hyper-params-scores-{}.csv".format(TRIAL))
    save_results_df(params_validation_df, "params-validation-df-{}.csv".format(TRIAL))
    # print(final_model_results)
    # trial = 56
    # final_model_results = pd.DataFrame.from_dict({"model": [], "score": [], "params": []})
    # model, trial = MDNetwork, 56
    # params, score = load_best_params(model, trial)
    # final_model_results = final_model_results.append({"model": model.__name__, "score": score, "params": params}, ignore_index=True)
    # save_results_df(final_model_results, "best-hyper-params-scores-{}.csv".format(trial))
    #
    # final_model_results = pd.DataFrame.from_dict({"model": [], "score": [], "params": []})
    # model, trial = XGBForecaster, 39
    # params, score = load_best_params(model, trial)
    # final_model_results = final_model_results.append({"model": model.__name__, "score": score, "params": params}, ignore_index=True)
    # save_results_df(final_model_results, "best-hyper-params-scores-{}.csv".format(trial))
