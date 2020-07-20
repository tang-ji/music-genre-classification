import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import yamnet.features as features_lib
import yamnet.params as params

from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.layers import *

from librosa.core import load
from librosa.feature import melspectrogram


def conv(layer, filters, kernel_size, pool_size):
    x = Conv2D(filters, kernel_size=kernel_size)(layer)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=pool_size)(x)
    return x

def createModelCNN(shape1, shape2):
    input = Input(shape=(shape1, shape2, ))
    input_reshape = Reshape((shape1, shape2, 1))(input)

    conv1 = conv(input_reshape, 64, (3,3), (2,2))
    conv2 = conv(conv1, 128, (3,3), (2,2))
    conv3 = conv(conv2, 256, (5,5), (2,2))
    # conv4 = conv(conv3, 512, (3,3), (4,4))

    flatten = Flatten()(conv3)
    dense1 = Dropout(0.5)(Dense(1024)(flatten))
    dense1 = keras.layers.LeakyReLU(alpha=0.1)(dense1)
    dense2 = Dropout(0.5)(Dense(256)(dense1))
    dense2 = keras.layers.LeakyReLU(alpha=0.1)(dense2)

    output = Dense(10, activation="softmax")(dense2)

    return Model(input, output)


class ModelCNN():
    def __init__(self, genres_list=None):
        self.model = None
        if genres_list is None:
            self.genres_list = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
        else:
            self.genres_list = genres_list
        
    def build(self, shape1=128, shape2=128):
        self.model = createModelCNN(shape1, shape2)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
        
    def predict_music(self, music_path):
        y, sr       = load(music_path, mono=True)
        spectrogram = features_lib.waveform_to_log_mel_spectrogram(tf.squeeze([y], axis=0), params)
        patches = features_lib.spectrogram_to_patches(spectrogram, params)
        x_pred = np.stack(data_chunks)
        y_pred = np.mean(self.model.predict(x_pred), axis=0)
        for i in range(len(self.genres_list)):
            print("{:10s} {:6.2f}%".format(self.genres_list[i], y_pred[i]*100))
        
    def save(self, path='model/'):
        self.model.save(path + 'CNN_model')
             
    def load(self, path='model/'):
        self.model = keras.models.load_model(path + 'CNN_model')
