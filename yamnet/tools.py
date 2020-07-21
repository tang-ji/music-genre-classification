import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
import matplotlib.pyplot as plt

import features as features_lib
import params as params
import yamnet as yamnet_model
import tensorflow.compat.v1 as tf

from librosa.core import load
from librosa.feature import melspectrogram

def draw_prediction(class_names, scores, figsize=(10, 4), p=0.3):
    mean_scores = np.mean(scores, axis=0)
    top_N = 5
    top_class_indices = np.argsort(mean_scores)[::-1][:top_N]
    plt.figure(figsize=figsize, dpi=100)
    plt.imshow(scores[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r')
    # Compensate for the PATCH_WINDOW_SECONDS (0.96 s) context window to align with spectrogram.
    patch_padding = (params.PATCH_WINDOW_SECONDS / 2) / p
    plt.xlim([-patch_padding, scores.shape[0] + patch_padding])
    # Label the top_N classes.
    yticks = range(0, top_N, 1)
    plt.yticks(yticks, [class_names[top_class_indices[x]] for x in yticks])
    _ = plt.ylim(-0.5 + np.array([top_N, 0]))

def fine_tuning_model(model_yamnet):
    layer_input = layers.Input(shape=[params.PATCH_FRAMES, params.PATCH_BANDS])
    x = layer_input
    for layer in model_yamnet.layers[63:-2]:
        x = layer(x)
    predictions = model_yamnet.layers[-1](model_yamnet.layers[-2](x))

    logits2 = layers.Dense(units=params.NUM_CLASSES2, use_bias=True)(x)
    predictions2 = layers.Activation(
        name=params.EXAMPLE_PREDICTIONS_LAYER_NAME2,
        activation="softmax")(logits2)

    model_train = Model(layer_input, predictions2)

    waveform = layers.Input(batch_shape=(1, None))
    spectrogram = features_lib.waveform_to_log_mel_spectrogram(
        tf.squeeze(waveform, axis=0), params)
    patches = features_lib.spectrogram_to_patches(spectrogram, params)
    model_prediction = Model(waveform, [Model(layer_input, predictions)(patches), spectrogram, model_train(patches)])
    
    return model_train, model_prediction

class modelYamnet:
    def __init__(self, sr=22050):
        self.sound_class_names = yamnet_model.class_names('yamnet/yamnet_class_map.csv')
        self.music_genre_names = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
        params.SAMPLE_RATE = sr
        self.model_yamnet = yamnet_model.yamnet_frames_model(params)
        self.model_yamnet.trainable = False
        self.model_train, self.model_prediction = fine_tuning_model(self.model_yamnet)
        self.last_prediction = None
    
    def save(self):
        self.model_prediction.save_weights("model/weights_general.h5")
        
    def load(self):
        self.model_prediction.load_weights("model/weights_general.h5")
            
    def predict_music(self, wav_file_name):
        wav_data, sr = load(wav_file_name)
        pred_sound, spectrogram, pred_music = self.model_prediction.predict(np.reshape(wav_data, [1, -1]), steps=1)
        self.last_prediction = [pred_sound, spectrogram, pred_music]
        return pred_sound, spectrogram, pred_music
    
    def draw_prediction(self, p=0.3, figsize=(10, 4)):
        if self.last_prediction is not None:
            draw_prediction(self.sound_class_names, self.last_prediction[0], p=p, figsize=figsize)
            draw_prediction(self.music_genre_names, self.last_prediction[2], p=p, figsize=figsize)
            
    def predict_draw(self, wav_file_name, p=0.3, figsize=(10, 4)):
        self.predict_music(wav_file_name)
        self.draw_prediction(p=p, figsize=figsize)
        