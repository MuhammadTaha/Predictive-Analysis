try:
    from src.forecaster.abstract_forecaster import *
except ModuleNotFoundError:
    from .abstract_forecaster import *
from sklearn.svm import SVR
import numpy as np

class LSTM(AbstractForecaster):
    params_grid = {
        "epsilon": list(range(0, 20, 2)),
        "gamma": ['auto'] + list(range(1, 100, 10))
    }

    def __init__(self):
        model = Sequential()
        # model.add(Embedding(max_features, 128))
        model.add(LSTM(64,
                       activation='relu',
                       # dropout=0.2,
                       # recurrent_dropout=0.2
                       ))
        model.add(Dense(1, activation='sigmoid'))


    def _decision_function(self, X):
        return self.svr.predict(X)

    def _train(self, data):
        X, y = data.all_train_data()
        self.svr.fit(X, y)

    def _build(self):
        pass

    def score(self, X, y):
        prediction = self.predict(X)
        return np.sqrt(np.mean(np.square((prediction - y + 0.1) / (y + 0.1))))

    def save(self):
        file_name = self.__class__.__name__ + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
        params = self.svr.get_params()
        joblib.dump(params, os.path.join(MODEL_DIR, file_name))
        return os.path.join(MODEL_DIR, file_name)

    @staticmethod
    def load_model(file_name):
        model = SVRForecaster()
        model.svr.set_params(joblib.load(os.path.join(MODEL_DIR, file_name)))
        return model

# # from .data import DataExtraction
# # from .data_extraction import *
# # import pandas as pd
# # import zipfile
# # import os
# # import tensorflow as tf
# # import numpy as np
# # import datetime
# # # import random
# # from keras import keras.layer
# #
# # class LSTM(DataExtraction):
# #     def __init__(self):
# #         print("in lstm")
#
#
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
#
# from sklearn.model_selection import train_test_split
# import numpy as np
# import matplotlib.pyplot as plt
#
# from src.data import abstract_data
#

import sys
sys.path.append("..")


import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt
import tensorflow as tf

from .abstract_data import AbstractData

# tf.enable_eager_execution()


#
class LSTMForecaster:
    def forecastor(self):
#
        # data = [[i for i in range(100)]]
        # data = np.array(data, dtype=float)
#         target = [[i for i in range(1, 101)]]
#         target = np.array(target, dtype=float)
#
        # data = data.reshape((1, 1, 100))
#         target = target.reshape((1, 1, 100))
#         x_test = [i for i in range(100, 200)]
#         x_test = np.array(x_test).reshape((1, 1, 100));
#         y_test = [i for i in range(101, 201)]
#         y_test = np.array(y_test).reshape(1, 1, 100)
#
        # model = Sequential()
        # model.add(LSTM(100, input_shape=(1, 100), return_sequences=True))
        # model.add(Dense(100))
        # model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
        # model.fit(data, target, nb_epoch=10000, batch_size=1, verbose=2, validation_data=(x_test, y_test))
        #
        # predict = model.predict(data)
        #
        # plt.scatter(range(100),predict,c='r')
        # plt.scatter(range(100),y_test,c='g')
        # plt.show()
        #
        # plt.plot(history.history['loss'])
        # plt.show()

        print("in lstm")
        epochs_number = 4
        model = Sequential()
        # model.add(Embedding(max_features, 128))
        model.add(LSTM(64,
                        activation = 'relu',
                       # dropout=0.2,
                       # recurrent_dropout=0.2
                       ))
        model.add(Dense(1, activation='sigmoid'))

        # try using different optimizers and different optimizer configs
        # model.compile(loss='binary_crossentropy',
        #               optimizer='adam',
        #               metrics=['accuracy'])


        abData  = AbstractData()

        data = abData.get_training_data(store_id=[25])

        x = np.array(data)
        print("train batch shape", x.shape)
        import pdb; pdb.set_trace()
        #
        # print(train_data)
        # print(test_data)

        # model.fit(abData.get_training_data(epoch = epochs_number,store_id = [23]),
        #
        #           # batch_size=batch_size,
        #           epochs=epochs_number)
        # print(train_data)


        print('Train...')

        for store_data in data:
            train_data_array,test_data_array = store_data
            for train_data in train_data_array:
                print(np.array(train_data).shape)
                train_data = np.array(train_data).reshape(1,50,27)
                print(train_data)
                #  train,test = train_data
                # print(train_data)
                # print(test)
                # model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
                # history = model.fit(train_data)


        # model.summary()
        # print(history)
        # model.fit(train_data,epochs=epochs_number)


        # history = model.fit(X_train, y_train, validation_split=0.2, verbose=0)

        # history = model.fit(X_train,y_train)

        #
        # model.summary()
        # print(history)
        # score, acc = model.evaluate(test_data,
        #                             batch_size=50)
        #
        # score, acc = model.evaluate(test_data,
        #                             batch_size=batch_size)


        # plot_history(history)

        # print('Test score:', score)
        # print('Test accuracy:', acc)


    def plot_history(history):
      plt.figure()
      plt.xlabel('Epoch')
      plt.ylabel('Mean Abs Error [1000$]')
      plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
               label='Train Loss')
      plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
               label = 'Val loss')
      plt.legend()
      plt.ylim([0,5])