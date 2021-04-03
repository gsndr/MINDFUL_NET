from __future__ import print_function
import numpy as np

# random seeds must be set before importing keras & tensorflow
my_seed = 12
np.random.seed(my_seed)
import random

random.seed(my_seed)
import tensorflow as tf


tf.random.set_seed(12)

from hyperopt import Trials, STATUS_OK, tpe
from keras.layers import Input, Dense, Dropout, Flatten, Conv1D
from keras.models import Model
from keras.utils import np_utils

from keras import callbacks
from hyperas import optim
from hyperas.distributions import choice, uniform

from sklearn.metrics import confusion_matrix

from keras.optimizers import Adam
import global_config
from sklearn.model_selection import train_test_split
import time

def getResult(cm):
    tp = cm[0][0]  # attacks true
    fn = cm[0][1]  # attacs predict normal
    fp = cm[1][0]  # normal predict attacks
    tn = cm[1][1]  # normal as normal
    attacks = tp + fn
    normals = fp + tn
    OA = (tp + tn) / (attacks + normals)
    AA = ((tp / attacks) + (tn / normals)) / 2
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F1 = 2 * ((P * R) / (P + R))
    FAR = fp / (fp + tn)
    TPR = tp / (tp + fn)
    r = (tp, fn, fp, tn, OA, AA, P, R, F1, FAR, TPR)
    return r

def data():





    y_train = global_config.train_Y
    y_test = global_config.test_Y
    x_train = global_config.train_X
    x_test = global_config.test_X

    nb_classes = 2
    y_train = np_utils.to_categorical(y_train, nb_classes)
    #y_test = np_utils.to_categorical(y_test, nb_classes)
    return x_train, y_train, x_test, y_test


def getBatchSize(p, bs):
    return bs[p]


def CNN(x_train, y_train, x_test, y_test):
    input_shape = (x_train.shape[1], x_train.shape[2])
    print(input_shape)
    input2 = Input(input_shape)

    l1 = Conv1D(64, kernel_size=1, activation='relu', name='conv0', kernel_initializer='glorot_uniform')(
        input2)
    l1 = Dropout({{uniform(0, 1)}})(l1)

    l1 = Flatten()(l1)

    l1 = Dense(320, activation='relu', kernel_initializer='glorot_uniform')(
        l1)
    # l1= BatchNormalization()(l1)

    l1 = Dropout({{uniform(0, 1)}})(l1)

    l1 = Dense(160, activation='relu', kernel_initializer='glorot_uniform')(
        l1)
    l1 = Dropout({{uniform(0, 1)}})(l1)

    softmax = Dense(2, activation='softmax', kernel_initializer='glorot_uniform')(l1)

    adam = Adam(lr={{uniform(0.0001, 0.01)}})
    model = Model(inputs=input2, outputs=softmax)
    #model.summary()
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer=adam)

    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10,
                                restore_best_weights=True),
    ]

    XTraining, XValidation, YTraining, YValidation = train_test_split(x_train, y_train, stratify=y_train,test_size=0.2)  # before model building
    tic = time.time()
    h = model.fit(x_train, y_train,
                        batch_size={{choice([32, 64, 128, 256, 512])}},
                        epochs=150,
                        verbose=2,
                        callbacks=callbacks_list,
                        validation_data=(XValidation,YValidation))

    toc =time.time()

    scores = [h.history['val_loss'][epoch] for epoch in range(len(h.history['loss']))]
    score = min(scores)
    print('Score', score)
    print(x_test.shape)
    predictions = model.predict(x_test, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    cmTest = confusion_matrix(y_test, y_pred)
    global_config.savedScore.append(cmTest)
    predictions = model.predict(XValidation, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    val = np.argmax(YValidation, axis=1)
    print(val)
    cmTrain = confusion_matrix(val, y_pred)
    global_config.savedTrain.append(cmTrain)



    print('Best score', global_config.best_score)

    if global_config.best_score > score:
        global_config.best_score = score
        global_config.best_model = model
        global_config.best_numparameters = model.count_params()
        global_config.best_time = toc - tic

    return {'loss': score, 'status': STATUS_OK, 'n_epochs': len(h.history['loss']),
            'n_params': model.count_params(), 'model': global_config.best_model, 'time': toc - tic}








def hypersearch(train_X1, train_Y1, test_X1, test_Y1, modelName, testPath):
    trials = Trials()
    global_config.train_X = train_X1
    global_config.train_Y = train_Y1
    global_config.test_X = test_X1
    global_config.test_Y = test_Y1
    global_config.best_model = None
    global_config.best_score=np.inf
    bs = [32, 64, 128, 256, 512]
    best_run, best_model = optim.minimize(model=CNN,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=2,
                                          trials=trials)
    outfile = open(testPath+'MINDFUL.csv', 'w')
    outfile.write("\nHyperopt trials")

    outfile.write(
        "\ntid , epochs,loss , learning_rate , Dropout , Dropout1,batch_size, TP_VAL, FN_VAL, FP_VAL, TN_VAL, OA_VAL, P_VAL, R_VAL, F1_VAL, TP_TEST, FN_TEST, FP_TEST, TN_TEST, OA_TEST,P_TEST, R_TEST, F1_TEST")
    for trial, test, train in zip(trials.trials, global_config.savedScore, global_config.savedTrain):
        t = getResult(test)
        v = getResult(train)
        print(type(trial['misc']['vals']['Dropout_1']))

        outfile.write(
            "\n%s , %s, %f , %f , %s , %f, %s, %d , %d , %d, %d, %f , %f , %f, %f ,%d , %d , %d, %d, %f, %f, %f, %f" % (
            trial['tid'],
            trial['result']['n_epochs'],
            trial['result']['loss'],
            trial['misc']['vals']['lr'][0],
            trial['misc']['vals']['Dropout'][0],
            trial['misc']['vals']['Dropout_1'][0],
            getBatchSize(trial['misc']['vals']['batch_size'][0], bs),
            v[0], v[1], v[2], v[3], v[4], v[6], v[7], v[8],
            t[0], t[1], t[2], t[3], t[4], t[6], t[7], t[8]
            ))

    outfile.write('\nBest model:\n ')
    outfile.write(str(best_run))

    global_config.best_model.save(modelName)
    return global_config.best_model, global_config.best_time
