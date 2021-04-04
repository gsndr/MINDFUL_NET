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
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import pandas as pd
import sys
sys.path.insert(1, 'MindfulNET')
from MindfulNET.MINDFUL  import MINDFUL_NET
from MindfulNET import Utils
import argparse






def main():
    # MIDFUL.py -i MINDFUL.conf -d AAGM17
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--input_file', action='store', dest='input_file',
                             help='Path Configuration File')


    parser.add_argument('-d', '--dataset_name', action='store', dest='dataset_name', help='Dataset name', required=True)
    args = parser.parse_args()
    dataset = args.dataset_name
    if args.input_file != None:
        config = configparser.ConfigParser()
        config.read(args.input_file)
    else:
        print('Error: insert a path of the configuration file')



    dsConf = config[dataset]
    pathModels=dsConf.get('pathModels')
    pathDs = dsConf.get('pathDataset')
    train = pd.read_csv(pathDs+'Train_OneClsNumeric.csv')
    print(train.head(8))
    cls = dsConf.get('label')
    X, Y = Utils.getXY(train, cls)

    #MINDFUL
    clf=MINDFUL_NET(dsConf)
    clf.fit(X,Y)

    #Test prediction
    test=pd.read_csv(pathDs+'Test_OneClsNumeric.csv')
    X_test, Y_test= Utils.getXY(test, cls)

    Y_pred=clf.predict(X_test)


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