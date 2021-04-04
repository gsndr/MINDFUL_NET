import configparser
import numpy as np

from sklearn.metrics import confusion_matrix

np.random.seed(12)
import tensorflow

tensorflow.random.set_seed(12)
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
session = InteractiveSession(config=config)
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import sys

sys.path.insert(1, 'MindfulNET')
from MindfulNET.MINDFUL import MINDFUL_NET
from MindfulNET import Utils
import argparse


def main():
    dsConf = {'pathModels': 'models/AAGM17/', 'testName': 'AAGM17'}
    pathModels = dsConf.get('pathModels')
    pathDs = 'datasets/AAGM17/'
    train = pd.read_csv(pathDs + 'Train_OneClsNumeric.csv')
    print(train.head(8))
    cls = 'classification'

    # MINDFUL
    clf = MINDFUL_NET(dsConf, autoencoderA=pathModels + 'autoencoderAttacks.h5',
                      autoencoderN=pathModels + 'autoencoderNormal.h5', model=pathModels+'MINDFUL.h5')


    # Test prediction
    test = pd.read_csv(pathDs + 'Test_OneClsNumeric.csv')
    X_test, Y_test = Utils.getXY(test, cls)

    Y_pred = clf.predict(X_test)

    cm = confusion_matrix(Y_test, Y_pred)
    print('Prediction Test')
    print(cm)
    # create pandas for results
    columns = ['TP', 'FN', 'FP', 'TN', 'OA', 'AA', 'P', 'R', 'F1', 'FAR(FPR)', 'TPR']
    results = pd.DataFrame(columns=columns)
    r = Utils.getResult(cm, 2)

    dfResults = pd.DataFrame([r], columns=columns)
    print(dfResults)

    results = results.append(dfResults, ignore_index=True)

    results.to_csv('AAGM_results.csv', index=False)


if __name__ == "__main__":
    main()
