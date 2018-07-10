import argparse

from forecaster.XGBForecaster import XGBForecaster
from data.feedforward_data import Data

# add list of models here
MODEL_LIST = [XGBForecaster(), ]


def training_task():
    try:
        data = Data.load_data()
    except:
        data = Data()
        data.save()
    scores = []
    for model in MODEL_LIST:
        model_name = type(model).__name__
        print("fitting model {}".format(model_name))
        model.fit(data)
        print("validating model {}".format(model_name))
        scores.append(model.score(data))
        print("model score {}".format(scores[-1]))
        model.save()
    best_score_idx = scores.index(max(scores))
    print("best scoring model {}".format(type(MODEL_LIST[best_score_idx]).__name__))


if __name__ == '__main__':
    training_task()
