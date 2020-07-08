import keras
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.layers import *


def conv(layer, filters, kernel_size, pool_size):
    x = Conv2D(filters, kernel_size=kernel_size)(layer)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=pool_size)(x)
    return x

def createModelCNN():
    input = Input(shape=(128, 128, ))
    input_reshape = Reshape((128, 128, 1))(input)

    conv1 = conv(input_reshape, 64, (3,3), (2,2))
    conv2 = conv(conv1, 128, (3,3), (2,2))
    conv3 = conv(conv2, 256, (5,5), (2,2))
    conv4 = conv(conv3, 512, (3,3), (4,4))

    flatten = Flatten()(conv4)
    dense1 = Dropout(0.5)(Dense(1024)(flatten))
    dense1 = keras.layers.LeakyReLU(alpha=0.1)(dense1)
    dense2 = Dropout(0.5)(Dense(256)(dense1))
    dense2 = keras.layers.LeakyReLU(alpha=0.1)(dense2)

    output = Dense(10, activation="softmax")(dense2)

    return Model(input, output)


class ModelCNN():
    def __init__(self):
        self.model = None
        
    def build(self):
        self.model = createModelCNN()
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
        
    def predict_music(self, music_path):
        y, sr       = load(music_path, mono=True)
        S           = melspectrogram(y, sr).T
        S           = S[:-1 * (S.shape[0] % 128)]
        num_chunk   = S.shape[0] / 128
        data_chunks = np.split(S, num_chunk)
        x_pred = np.stack(data_chunks)
        y_pred = np.mean(self.model.predict(x_pred), axis=0)
        for i in range(len(genres_list)):
            print("{:10s} {:6.2f}%".format(genres_list[i], y_pred[i]*100))
        
    def save(self, path='model/'):
        self.model.save(path + 'CNN_model')
             
    def load(self, path='model/'):
        self.model = keras.models.load_model(path + 'CNN_model')
