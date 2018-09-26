try:
    from .abstract_forecaster import *
except ModuleNotFoundError:
    from src.forecaster.abstract_forecaster import *


import uuid

import numpy as np
from keras.layers import Dense, LSTM
from keras.models import Sequential

class LSTMForecaster(AbstractForecaster):
    params_grid = {
        "hidden_units": list(range(30, 200, 10)),
        "dropout": [0, 0.2, 0.4, 0.6, 0.8],
        "recurrent_dropout": [0, 0.2, 0.4, 0.6, 0.8],
        "num_timesteps": list(range(1, 15, 2)),
    }

    def __init__(self, num_timesteps, features_count, hidden_units=64, dropout=0, recurrent_dropout=0):
        self.num_timesteps, self.features_count, self.hidden_units, self.dropout, self.recurrent_dropout = \
            num_timesteps, features_count, hidden_units, dropout, recurrent_dropout

        model = Sequential()
        # model.add(Embedding(max_features, 128))
        model.add(LSTM(hidden_units,
                       activation='relu',
                       dropout=dropout,
                       recurrent_dropout=recurrent_dropout,
                       input_shape=(num_timesteps, features_count)
                       ))
        model.add(Dense(1, activation='linear'))

        model.compile(loss=tf_rmspe, optimizer='adam')

        self.model = model

    def _decision_function(self, X):
        return self.model.predict(X)

    def _train(self, data):
        data.reset_timesteps_per_point(self.num_timesteps)

        os.makedirs(".temp", exist_ok=True)
        name = self.__class__.__name__ + str(uuid.uuid4())[:5]

        print("({}) Start training".format(self.__class__.__name__))
        #  linear regression can't overfit, so as stopping criterion we take that the changes are small

        train_losses, val_losses, val_times = [np.inf], [np.inf], [np.inf]
        train_step = 0

        while early_stopping(train_step, val_losses, data.epochs):  # no improvement in the last 10 epochs
            X, y = data.next_train_batch()
            # run train step
            train_loss = self.model.train_on_batch(X, y, sample_weight=X[:, -1])

            #  logging.info("({}) Step {}: Train loss {}".format(self.__class__.__name__, train_step, train_loss))
            print("({}) Step {}: Train loss {}".format(self.__class__.__name__, train_step, train_loss))
            train_losses.append(train_loss)

            if data.is_new_epoch or train_step % 100 == 0:
                val_loss = self.model.evaluate(data.X_val, data.y_val, sample_weight=data.X_val[:, -1])
                val_losses.append(val_loss)

                if val_loss == min(val_losses[1:]):
                    self.save_model(".temp/{}_params".format(name))
                    print("saved")

                val_times.append(train_step)
                print("({}) Step {}: Val loss {}".format(self.__class__.__name__, train_step, val_loss))
            train_step += 1

        self.restore_model(".temp/{}_params".format(name))
        print("({}) Finished with val loss {}".format(self.__class__.__name__, min(val_losses)))

    def _build(self):
        pass

    def score(self, X, y):
        prediction = self.predict(X)
        return rmspe(y, prediction)

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save_weights(os.path.join(path, 'model.h5'))

    def restore_model(self, path):
        self.model.load_weights(os.path.join(path, 'model.h5'))

    def save(self):
        file_name = self.__class__.__name__ + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
        path = os.path.join(MODEL_DIR, file_name)
        self.restore_model(path)
        return path

    @staticmethod
    def load_model(file_name):
        lstm = LSTM()
        lstm.restore_model(os.path.join(MODEL_DIR, file_name))
        return lstm
