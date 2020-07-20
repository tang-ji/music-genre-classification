import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf

import yamnet.features as features_lib
import yamnet.params as params

from librosa.core import load
from librosa.feature import melspectrogram
from librosa import power_to_db
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import shuffle

RAW_DATAPATH="genres_raw"
class Label():
    def __init__(self, genres_list):
        self.le         = LabelEncoder()
        integer_encoded = self.le.fit_transform(genres_list)
        self.one        = OneHotEncoder()
        self.one.fit(integer_encoded.reshape(len(integer_encoded), 1))
        
    def transform(self, Y):
        Y = self.one.transform(np.expand_dims(self.le.transform(Y), axis=1))
        return Y

class Data():
    def __init__(self, genres):
        self.raw_data = None
        self.genres_list = genres
        self.train_set  = None
        self.test_set   = None
        self.label = Label(self.genres_list)
        

    def load_data(self, datapath):
        self.datapath = datapath
        records = list()
        for i, genre in enumerate(self.genres_list):
            GENREPATH = self.datapath + genre + "/"
            for j, track in enumerate(os.listdir(GENREPATH)):
                if j>10:
                    break
                TRACKPATH   = GENREPATH + track
                print("%d.%s\t\t%s (%d)" % (i + 1, genre, TRACKPATH, j + 1))
                y, sr       = load(TRACKPATH, mono=True)
                spectrogram = features_lib.waveform_to_log_mel_spectrogram(tf.squeeze([y], axis=0), params)
                patches = features_lib.spectrogram_to_patches(spectrogram, params)
                data_chunks = [(data, genre) for data in patches]
                records.append(data_chunks)

        records = [data for record in records for data in record]
        self.raw_data = pd.DataFrame.from_records(records, columns=['spectrogram', 'genre'])
      
    def build_dataset(self):
        df = self.raw_data.copy()
        df = shuffle(df)

        train_records, test_records = list(), list()
        for i, genre in enumerate(self.genres_list):
            genre_df    = df[df['genre'] == genre]
            n = round(len(genre_df) * 0.9)
            train_records.append(genre_df.iloc[:n].values)
            test_records.append(genre_df.iloc[n:].values)

        train_records   = shuffle([record for genre_records in train_records    for record in genre_records])
        test_records    = shuffle([record for genre_records in test_records     for record in genre_records])

        self.train_set  = pd.DataFrame.from_records(train_records,  columns=['spectrogram', 'genre'])
        self.test_set   = pd.DataFrame.from_records(test_records,   columns=['spectrogram', 'genre'])
        
    def get_train_set(self):
        x_train = np.stack(self.train_set['spectrogram'].values)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
        y_train = np.stack(self.train_set['genre'].values)
        y_train = self.label.transform(y_train)
        print("x_train shape: ", x_train.shape)
        print("y_train shape: ", y_train.shape)
        return x_train, y_train

    def get_test_set(self):
        x_test  = np.stack(self.test_set['spectrogram'].values)
        x_test  = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))
        y_test  = np.stack(self.test_set['genre'].values)
        y_test  = self.label.transform(y_test)
        print("x_test shape : ", x_test.shape)
        print("y_test shape : ", y_test.shape)
        return x_test, y_test

    def save(self):
        with open(RAW_DATAPATH, 'wb') as outfile:
            pickle.dump(self.raw_data, outfile, pickle.HIGHEST_PROTOCOL)
        print('-> Data() object is saved.\n')
        return

    def load(self):
        with open(RAW_DATAPATH, 'rb') as infile:
            self.raw_data   = pickle.load(infile)
        print("-> Data() object is loaded.")
        return