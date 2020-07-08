import keras
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.layers import *

def conv(layer, filters, kernel_size, pool_size):
    x = Conv2D(filters, kernel_size=kernel_size, activation="relu")(layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=pool_size)(x)
    return x

def createModelCNN():
    input = Input(shape=(128, 128, ))
    input_reshape = Reshape((128, 128, 1))(input)

    conv1 = conv(input_reshape, 64, (3,3), (2,2))
    conv2 = conv(conv1, 128, (3,3), (2,2))
    conv3 = conv(conv2, 256, (3,3), (4,4))
    conv4 = conv(conv3, 512, (3,3), (4,4))

    flatten = Flatten()(conv4)
    dense1 = Dropout(0.5)(Dense(1024)(flatten))
    dense2 = Dropout(0.5)(Dense(256)(dense1))

    output = Dense(10, activation="softmax")(dense2)

    return Model(input, output)

class ModelCNN:
    def __init__():
        self.model = None
        
    def build():
        self.model = createModelCNN()
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
        
    def save(path='model/'):
        self.model.save(path + 'CNN_model')
             
    def load(path='model/'):
        self.model = keras.models.load_model(path + 'CNN_model')

