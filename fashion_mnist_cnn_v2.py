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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
    // Model hyper-parameters
    x1 = 32-128 filters in each CNN layer
    x2 = 50-100 neurons in first dense layer
    x3 = no dropout, 0.1 dropout, 0.2 dropout
    x4 = learning rate 0.005 or 0.05 
    x5 = kernel size cnn 2-4
    
    // Data augmentation hyper-parameters
    x6 = rotation_range: 0-30
    x7 = width_shift_range: 0.0-0.3,
    x8 = height_shift_range: 0.0-0.3
    x9 = horizontal_flip: 0 or 1
    
    We will use a genetic algorithm to find the best combinations of these hyperparameters. 
    The fitness-function is based on model accuracy on test-dataset. 
    
    
    Best so far with 92.29 % 
        0.6151899027462715,
        0.8630441247971613,
        0.48568135841046134,
        0.0032114426394213025,
        0.30784428242021955,
        0.1473845685726038,
        0.3145443359129966,
        0.07676705567952473,
        0.4287397570066659
      ]
"""

def define_model_ga(hyperparameter_arr):
    # set hyperparameters based on array pop
    neurons_cnn = 32 + int((128-32+1) * hyperparameter_arr[0]) # min 32, max 128
    neurons_first = 50 + int((100-50+1)* hyperparameter_arr[1])
    dropout = (0.3)* hyperparameter_arr[2] 
    learning_rate = 0.005 + (0.05-0.005) * hyperparameter_arr[3]
    kernel_size_cnn = 2 + int((4-2+1)*hyperparameter_arr[4])
    
    # define model
    model = Sequential()
    model.add(Conv2D(neurons_cnn, (kernel_size_cnn, kernel_size_cnn), activation='relu', kernel_initializer="he_uniform", input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(neurons_cnn, (kernel_size_cnn, kernel_size_cnn), activation='relu', kernel_initializer="he_uniform", input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(neurons_first, activation='relu'))
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
            # set data augmentation parameters
            rotation_range = 30*arr[5]
            width_shift = 0.3*arr[6]
            height_shift = 0.3*arr[7]
            horizontal_flip = bool(int(2*arr[8]))
            # do data augmentation 
            datagen = ImageDataGenerator(
                rotation_range=rotation_range,
                width_shift_range=width_shift,
                height_shift_range=height_shift,
                horizontal_flip=horizontal_flip,
                validation_split=0.2)
            # define model
            model = define_model_ga(arr)
            callback = tf.keras.callbacks.EarlyStopping(patience=5)
            # fit model
            history = model.fit(datagen.flow(trainX, trainY, batch_size=32,subset='training'), 
                      validation_data=datagen.flow(trainX, trainY,batch_size=8, subset='validation'), 
                      epochs=50,callbacks=[callback], verbose=1) 
            #history = model.fit(trainX, trainY, epochs=20, batch_size=32, validation_split = 0.2, verbose=1, callbacks=[callback])
            # evaluate model
            _, acc = model.evaluate(testX, testY, verbose=0)
            pop_scores.append(acc)
            print('Accuracy pop: > %.3f' % (acc * 100.0), arr)
            
    return pop_scores