import numpy as np

my_seed = 12
np.random.seed(my_seed)
import random

random.seed(my_seed)


import tensorflow
tensorflow.random.set_seed(12)

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow.keras.models import load_model
from keras import backend as K
np.set_printoptions(suppress=True)
from MindfulNET import AutoencoderHypersearch as ah, CNNHypersearch as ch


class MINDFUL_NET():
    def __init__(self, dsConfig, autoencoderA=None, autoencoderN=None, model=None):
        self.ds = dsConfig
        fileOutput = self.ds.get('pathModels') + 'result' + self.ds.get('testPath') + '.txt'
        self.file = open(fileOutput, 'w')
        self.file.write('Result time for: \n' )
        self.file.write('\n')
        self.pathModels =self.ds.get('pathModels')
        if autoencoderA!=None:
            self.autoencoderA=load_model(autoencoderA)
        else:
            self.autoencoderA=None
        if autoencoderN!=None:
            self.autoencoderN=load_model(autoencoderN)
        else:
            self.autoencoderN=None
        if model!=None:
            self.model=load_model(model)
        else:
            self.model=None



    def createImage(self, train_X, trainN, trainA):
        rows = [train_X, trainN, trainA]
        rows = [list(i) for i in zip(*rows)]

        train_X = np.array(rows)

        if K.image_data_format() == 'channels_first':
            x_train = train_X.reshape(train_X.shape[0], train_X.shape[1], train_X.shape[2])
            input_shape = (train_X.shape[1], train_X.shape[2])
        else:
            x_train = train_X.reshape(train_X.shape[0], train_X.shape[2], train_X.shape[1])
            input_shape = (train_X.shape[2], train_X.shape[1])
        return x_train, input_shape
        
    def get_features(self,data):
        N_RE = self.autoencoderN.predict(data)
        A_RE= self.autoencoderA.predict(data)
        X_image, input_Shape = self.createImage(data, N_RE, A_RE)
        return X_image, input_Shape
        
    


    def predict(self,data):
        X_image, input_shape=self.get_features(data)
        predictions = self.model.predict(X_image)
        y_pred = np.argmax(predictions, axis=1)
        return y_pred

    def predict_proba(self, data):
        X_image, input_shape = self.get_features(data)
        y_prob = self.model.predict(X_image)
        return y_prob
        
    def get_model(self):
        return self.model


    def fit(self, X, Y):

        print('MINDFUL EXECUTION')
        pd.set_option('display.expand_frame_repr', False)

        index_normal = np.where(Y == 1)
        print(index_normal)
        train_XN=X[index_normal][:]
        print(X.shape)
        print(train_XN.shape)
        train_YN = Y[index_normal][:]
        index_anormal = np.where(Y == 0)[0]
        train_XA = X[index_anormal][:]
        train_YA = Y[index_anormal][:]

        print('Train data shape normal', train_XN.shape)
        print('Train target shape normal', train_YN.shape)


        print('Train data shape anormal', train_XA.shape)
        print('Train target shape anormal', train_YA.shape)


        if (self.autoencoderN ==None):

            self.autoencoderN, best_time, encoder = ah.hypersearch(train_XN, train_YN, train_XN, train_YN,
                                                             self.pathModels + 'autoencoderNormal.h5',self.pathModels+self.ds.get('testPath')+'Normal')

            self.file.write("Time Training Autoencoder Normal: %s" % best_time)
            self.file.write('\n')

        else:
            print("Load autoencoder Normal from disk")
            #self.autoencoderN = load_model(self.pathModels + 'autoencoderNormal.h5')
            self.autoencoderN.summary()


        train_RE = self.autoencoderN.predict(X)




        if (self.autoencoderA ==None):

            self.autoencoderA, best_time, encoder = ah.hypersearch(train_XA, train_YA, train_XA, train_XA,
                                                              self.pathModels + 'autoencoderAttacks.h5', self.pathModels+self.ds.get('testPath')+'Attack')
            self.file.write("Time Training Autoencoder Attacks: %s" % best_time)
            self.file.write('\n')


        else:
            print("Load autoencoder Attacks from disk")
            #self.autoencoderA = load_model(self.pathModels + 'autoencoderAttacks.h5')
            self.autoencoderA.summary()

        train_REA = self.autoencoderA.predict(X)
        #test_REA = self.autoencoderA.predict(test_X)



        train_X_image, input_Shape = self.createImage(X, train_RE, train_REA)
        #test_X_image, input_shape = self.createImage(test_X, test_RE, test_REA)



        if (self.model==None):
            self.model, best_time,  = ch.hypersearch(train_X_image, Y, train_X_image, Y,
                                                              self.pathModels + 'MINDFUL.h5', self.pathModels+ self.ds.get('testPath') )
            self.file.write("Time Training CNN: %s" % best_time)
            self.file.write('\n')

        else:
            print("Load softmax from disk")
            #self.model = load_model(self.pathModels + 'MINDFUL.h5')
            self.model.summary()
        
        return self.model

        

        
        



