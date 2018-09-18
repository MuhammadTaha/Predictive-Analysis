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