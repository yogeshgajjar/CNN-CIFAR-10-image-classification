# EE569 Homework Assignment #5 - Competition
# Date: 3rd May 2020
# Name: Yogesh Sanat Gajjar
# USCID: 5476153636
# Email: ygajjar@usc.edu

# Version of OS - Ubuntu 18.04.3 LTS
# IDE - Visual Studio Code 

# ------------------------------------------------------------------------
#     Problem 2. CIFAR-10 Competition 
# ------------------------------------------------------------------------
# STEPS TO RUN THE CODE 
# 1.  Open terminal and change the directory where the source code files are downloaded. 
# 2.  To train the model from scratch, use 'train' in the terminal before executing. If you want to load the pretrained model, use 'load' in the terminal before executing. 
# 3.  Execute using the command python3 CIFAR-10_CNN.py train/load 


import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Activation, Conv2D, GlobalAveragePooling2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop
from matplotlib import pyplot as plt
from keras.callbacks import CSVLogger
from keras.models import model_from_json
import tensorflow as tf
import numpy as np
import math 
from PIL import Image
from keras.models import model_from_json
import os
import time 
import sys 


def loadImages(labels):
    (Xtr, ytr), (Xte, yte) = cifar10.load_data()
    ytr = keras.utils.to_categorical(ytr, labels)
    yte = keras.utils.to_categorical(yte, labels)
    Xtr = Xtr.astype('float32')
    Xte = Xte.astype('float32')
    Xtr /= 255
    Xte /= 255
    return Xtr, Xte, ytr, yte 

def myCNNArchitecture(weights=None):
    mycnn = Sequential()

    mycnn.add(Conv2D(96, (3, 3), padding = 'same', input_shape=(32, 32, 3)))
    mycnn.add(Activation('relu'))
    mycnn.add(Dropout(0.2))
    mycnn.add(Conv2D(96, (3, 3), padding = 'same'))
    mycnn.add(Activation('relu'))
    mycnn.add(Conv2D(96, (3, 3), padding = 'same'))
    mycnn.add(Activation('relu'))
    mycnn.add(MaxPooling2D(pool_size=(3,3), strides=2))
    mycnn.add(Dropout(0.5))

    mycnn.add(Conv2D(192, (3, 3), padding = 'same'))
    mycnn.add(Activation('relu'))
    mycnn.add(Conv2D(192, (3, 3), padding = 'valid'))
    mycnn.add(Activation('relu'))
    mycnn.add(Conv2D(192, (3, 3), padding = 'same'))
    mycnn.add(Activation('relu'))
    mycnn.add(MaxPooling2D(pool_size=(3,3), strides=2))
    mycnn.add(Dropout(0.5))
  
    mycnn.add(Conv2D(192, (3, 3), padding = 'same'))
    mycnn.add(Activation('relu'))
    mycnn.add(Conv2D(192, (1, 1), padding = 'valid'))
    mycnn.add(Activation('relu'))
    mycnn.add(Conv2D(10, (1, 1), padding = 'valid'))

    mycnn.add(GlobalAveragePooling2D())
    mycnn.add(Activation('softmax'))
    
    if weights:
        mycnn.load_weights(weights)

    return mycnn

def dataVisualize(hist, dropout, learningRate, key, weightDecay, batchSize):
    xavier = 'False'
    plt.figure(1, figsize=(12,8))
    plt.plot(hist.history['accuracy'], color='red',linewidth=2)
    plt.plot(hist.history['val_accuracy'],color='blue', linewidth=2)
    plt.title("Epoch Accuracy Plot \n Dropout: "+str(dropout)+"| Learning Rate: "+str(learningRate)+"| Optimizers: "+key+
                " | Weight Decay:"+str(weightDecay)+"| Batch Size: "+str(batchSize)+"| Filter Weight(xavier_normal): "+str(xavier))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train Accuracy', 'Test Accuracy'], loc='upper left')
    plt.savefig("epoch-accuracy.png")
    plt.show()

    plt.figure(2, figsize=(12,8))
    plt.plot(hist.history['loss'], color='red', linewidth=2)
    plt.plot(hist.history['val_loss'], color='blue', linewidth=2)
    plt.title("Epoch Loss Plot \n Dropout: "+str(dropout)+"| Learning Rate: "+str(learningRate)+"| Optimizers: "+key+
                " | Weight Decay: "+str(weightDecay)+"| Batch Size: "+str(batchSize)+"| Filter Weight(xavier_normal): "+str(xavier))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train loss', 'Test loss'], loc='upper right')
    plt.savefig("epoch-loss.png")
    plt.show()

def modelDetails(totalEpochs, batchSize, weightDecay, learningRate, key):
    print("******* DETAILS OF THE MODEL *********\n")
    print("** Hyper - parameters detail **\n")
    print("Epochs : ", totalEpochs)
    print("Batch Size: ", batchSize)
    print("Weight Decay: ", weightDecay)
    print("Learning rate: ", learningRate)
    print("Optimizer: ", key,"\n")
   

def saveModel(mycnn):

    newmodel = mycnn.to_json()
    with open("myModel.json", "w") as json_file:
        json_file.write(newmodel)

    mycnn.save_weights("myModel.h5")
    print("Saved model to disk")

def loadModel(json_file='myModel.json', h5_model='myModel.h5'):
    # load json and create model
    json_file = open(json_file, 'r')
    myloaded_model = json_file.read()
    json_file.close()
    finalModel = model_from_json(myloaded_model)
    finalModel.load_weights(h5_model)
    print("Loaded model from disk")
    
    return finalModel

def testPhase(final, Xte, yte, batchSize, optim):

    final.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
    score = final.evaluate(Xte, yte, verbose=0)
    # print("%s: %.2f%%" % (final.metrics_names[1], score[1]*100))

    score, acc = final.evaluate(Xte, yte,
                                batch_size=batchSize)

    print("******** FINAL TEST ACCURACY **********")
    print('Test score:', score)
    print('Test accuracy: ', acc*100, '%')



def main():
    if len(sys.argv) == 2:
        choice = str(sys.argv[1])

    labels, batchSize, totalEpochs, learningRate, dropout, key = 10, 64, 200, 0.01, 0.5, "SGD"
    start_time = time.time()

    weightDecay = learningRate/totalEpochs
    Xtrain, Xtest, ytrain, ytest = loadImages(labels)
    mycnn = myCNNArchitecture()
    
    
    optim = SGD(lr=learningRate, decay=weightDecay , momentum=0.9, nesterov=True)
    mycnn.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

    datagen = ImageDataGenerator(
    rotation_range=10,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
    )
    datagen.fit(Xtrain)


    tf.keras.callbacks.CSVLogger("log.csv", separator=',', append=False)
    csv_logger = CSVLogger('training.csv')

    if choice == 'train':
        modelDetails(totalEpochs, batchSize, weightDecay, learningRate, key)

        print("---- Training starts ----\n ")

        # mycnnHistory = mycnn.fit(Xtrain, ytrain, validation_data=(Xtest, ytest), epochs=totalEpochs, batch_size=batchSize, verbose = 1, callbacks=[csv_logger])
        mycnnHistory = mycnn.fit_generator(datagen.flow(Xtrain, ytrain, batch_size=batchSize), steps_per_epoch = len(Xtrain) / batchSize, epochs=totalEpochs, validation_data=(Xtest, ytest), verbose=0, callbacks=[csv_logger])

        print("---- Training ends ----\n")
        print("Average time taken by each epoch for training is: ", round(time.time() - start_time, 2)/totalEpochs, 's')
        print("Total time taken for training 50,000 images: ", (round(time.time() - start_time, 2))/60,'m\n')
        
        mycnn.summary()
        dataVisualize(mycnnHistory, dropout, learningRate, key, weightDecay, batchSize)
        saveModel(mycnn)
        final = loadModel()

    if choice == 'load':
        final = loadModel('myModel_load.json', 'myModel_load.h5')


    testPhase(final, Xtest, ytest, batchSize, optim)
    
if __name__ == "__main__":
    main()
