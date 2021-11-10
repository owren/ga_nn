# baseline cnn model for fashion mnist
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import SGD
import tensorflow as tf

import sys


device_name = "/gpu:0"


# load train and test dataset
def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()
    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm

"""
The target is to maximize the accuracy of the cnn-mnist model:
    We start by having a few decisions related to the model. 
    x1 = (32 or) 64 or 128 filters in each CNN layer
    x2 = 50 or 100 neurons in first dense layer
    x3 = 1 or 2 CNN layers
    x4 = he_uniform or random_normal
    x5 = no dropout, (0.1 dropout,) 0.2 dropout
    x6 = learning rate 0.005, (0.01) or 0.05 
    x7 = kernel size cnn (2,) 3, 4
    We will use a genetic algorithm to find the best combinations of these hyperparameters. 
    The fitness-function is based on model accuracy on test-dataset. 
"""

def define_model_ga(hyperparameter_arr):
    # set hyperparameters based on array pop
    neurons_cnn = [64, 128][hyperparameter_arr[0]]
    neurons_first = [50, 100][hyperparameter_arr[1]]
    cnn_layers = [1, 2][hyperparameter_arr[2]]
    initializer = ['he_uniform', 'random_normal'][hyperparameter_arr[3]]
    dropout = [False, 0.2][hyperparameter_arr[4]]
    learning_rate = [0.01, 0.05][hyperparameter_arr[5]]
    kernel_size_cnn = [3, 4][hyperparameter_arr[6]]
    
    # define model
    model = Sequential()
    for i in range(cnn_layers): 
        model.add(Conv2D(neurons_cnn, (kernel_size_cnn, kernel_size_cnn), activation='relu', kernel_initializer=initializer, input_shape=(28, 28, 1)))
        model.add(MaxPooling2D((2, 2)))
    if (dropout): 
        model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(neurons_first, activation='relu', kernel_initializer=initializer))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=learning_rate, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def evaluate_model_ga(pop): 
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # run model evaluation for each hyperparameter population
    pop_scores = []
    with tf.device(device_name):
        for arr in pop: 
            # define model
            model = define_model_ga(arr)
            callback = tf.keras.callbacks.EarlyStopping(patience=3)
            # fit model
            history = model.fit(trainX, trainY, epochs=20, batch_size=32, validation_split = 0.2, verbose=1, callbacks=[callback])
            # evaluate model
            _, acc = model.evaluate(testX, testY, verbose=0)
            pop_scores.append(acc)
            print('Accuracy pop: > %.3f' % (acc * 100.0), arr)
            
    return pop_scores