from __future__ import print_function
import numpy as np

from hyperopt import Trials, STATUS_OK, tpe
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform

np.random.seed(12)
import tensorflow
tensorflow.random.set_seed(12)


from keras.optimizers import Adam
from keras import callbacks
import time

import global_config
from sklearn.model_selection import train_test_split





def data():





    y_train = global_config.train_Y
    y_test = global_config.test_Y
    x_train = global_config.train_X
    x_test = global_config.test_X

    nb_classes = 2
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    return x_train, y_train, x_test, y_test


def getBatchSize(p, bs):
    return bs[p]


def Autoencoder(x_train, y_train, x_test, y_test):
    input_shape = (x_train.shape[1],)
    input2 = Input(input_shape)

    encoded = Dense(50, activation='relu',
                    kernel_initializer='glorot_uniform',
                    name='encod1')(input2)
    encoded = Dense(10, activation='relu',
                    kernel_initializer='glorot_uniform',
                    name='encod2')(encoded)
    encoded = Dropout({{uniform(0, 1)}})(encoded)
    decoded = Dense(50, activation='relu',
                    kernel_initializer='glorot_uniform',
                    name='decoder1')(encoded)
    decoded = Dense(x_train.shape[1], activation='linear',
                    kernel_initializer='glorot_uniform',
                    name='decoder3')(decoded)


    model = Model(inputs=input2, outputs=decoded)
    model.summary()

    adam=Adam(lr={{uniform(0.0001, 0.01)}})
    model.compile(loss='mse', metrics=['acc'],
                  optimizer=adam)
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10,
                                restore_best_weights=True),
    ]

    XTraining, XValidation, YTraining, YValidation = train_test_split(x_train, x_train, stratify=y_train,
                                                                      test_size=0.2)  # before model building
    tic = time.time()
    history = model.fit(XTraining, YTraining,
                        batch_size={{choice([32, 64, 128, 256, 512])}},
                        epochs=150,
                        verbose=2,
                        callbacks=callbacks_list,
                        validation_data=(XValidation, YValidation))

    toc = time.time()


    score = np.amin(history.history['val_loss'])
    print('Best validation loss of epoch:', score)


    scores = [history.history['val_loss'][epoch] for epoch in range(len(history.history['loss']))]
    score = min(scores)
    print('Score',score)
    predictions = model.predict(x_test)
    mse = np.nanmean(np.mean(np.power(x_test - predictions, 2), axis=1))

    print(mse)
    predictionsT = model.predict(XValidation)
    mseT = np.mean(np.mean(np.power(XValidation - predictionsT, 2), axis=1))
    print(mseT)
    global_config.autoScore.append(mse)
    global_config.autoTrain.append(mseT)


    print('Best score', global_config.best_score)




    if global_config.best_score > score:
        global_config.best_score = score
        global_config.best_model = model
        global_config.best_numparameters = model.count_params()
        global_config.best_time = toc - tic



    return {'loss': score, 'status': STATUS_OK, 'n_epochs': len(history.history['loss']), 'n_params': model.count_params(), 'model': global_config.best_model, 'time': toc - tic}


def hypersearch(train_X1, train_Y1, test_X1, test_Y1, pathModel, pathCSV):

    global_config.train_X =  train_X1
    global_config.train_Y =train_Y1
    global_config.test_X = test_X1
    global_config.test_Y =test_Y1
    global_config.best_model = None
    global_config.best_score = np.inf


    trials = Trials()

    bs = [32, 64, 128, 256, 512]
    best_run, best_model = optim.minimize(model=Autoencoder,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=2,
                                          trials=trials)
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    outfile = open(pathCSV+'Autoencoder.csv', 'w')
    outfile.write("\nHyperopt trials")

    outfile.write("\ntid , loss , learning_rate , epoch, Dropout , batch_size, time, mseTrain, mseTest")
    for trial, test, train in zip(trials.trials, global_config.autoScore, global_config.autoTrain):
        #outfile.write(str(trial))
        #print(test)

        outfile.write("\n%s , %f , %f , %s , %s, %s, %s, %f, %f" % (trial['tid'],
                                                                trial['result']['loss'],
                                                                trial['misc']['vals']['lr'][0],
                                                                trial['misc']['vals']['Dropout'],
                                                                trial['result']['n_epochs'],
                                                                getBatchSize(trial['misc']['vals']['batch_size'][0],
                                                                             bs),
                                                                trial['result']['time'],
                                                                train,
                                                                test
                                                                ))


    outfile.write('\nBest model:\n ')
    outfile.write(str(best_run))
    outfile.flush()
    global_config.best_model.save(pathModel)
    encoder = Model(inputs=global_config.best_model.input, outputs=global_config.best_model.get_layer('encod2').output)
    encoder.summary()

    return global_config.best_model, global_config.best_time, encoder
