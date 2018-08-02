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
from .abstract_data import AbstractData
#
class LSTMForecaster:
    def forecastor(self):
#
        data = [[i for i in range(100)]]
        data = np.array(data, dtype=float)
#         target = [[i for i in range(1, 101)]]
#         target = np.array(target, dtype=float)
#
        data = data.reshape((1, 1, 100))
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
        epochs_number = 4
        model = Sequential()
        # model.add(Embedding(max_features, 128))
        model.add(LSTM(128,activation = 'relu',
                       dropout=0.2,
                       recurrent_dropout=0.2
                       ))
        model.add(Dense(1, activation='sigmoid'))

        # try using different optimizers and different optimizer configs
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        print('Train...')

        abData  = AbstractData()

        model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

        model.fit(abData.get_training_data(epoch = epochs_number,store_id = [23]),

                  # batch_size=batch_size,
                  epochs=epochs_number)





        score, acc = model.evaluate(x_test, y_test,
                                    batch_size=batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)