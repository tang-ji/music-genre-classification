import numpy as np
from librosa.core import load
import soundfile as sf
import IPython.display as ipd

import matplotlib.pyplot as plt

import params
from tools import *
import yamnet as yamnet_model
import tensorflow.compat.v1 as tf

def draw_result(waveform, class_names, scores, spectrogram):

    # Visualize the results.
    plt.figure(figsize=(10, 8), dpi=100)

    # Plot the waveform.
    plt.subplot(3, 1, 1)
    plt.plot(waveform)
    plt.xlim([0, len(waveform)])
    # Plot the log-mel spectrogram (returned by the model).
    plt.subplot(3, 1, 2)
    plt.imshow(spectrogram.T, aspect='auto', interpolation='nearest', origin='bottom')

    # Plot and label the model output scores for the top-scoring classes.
    mean_scores = np.mean(scores, axis=0)
    top_N = 10
    top_class_indices = np.argsort(mean_scores)[::-1][:top_N]
    plt.subplot(3, 1, 3)
    plt.imshow(scores[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r')
    # Compensate for the PATCH_WINDOW_SECONDS (0.96 s) context window to align with spectrogram.
    patch_padding = (params.PATCH_WINDOW_SECONDS / 2) / params.PATCH_HOP_SECONDS
    plt.xlim([-patch_padding, scores.shape[0] + patch_padding])
    # Label the top_N classes.
    yticks = range(0, top_N, 1)
    plt.yticks(yticks, [class_names[top_class_indices[x]] for x in yticks])
    _ = plt.ylim(-0.5 + np.array([top_N, 0]))
    
class modelYamnet:
    def __init__(self, sr=22050):
        self.class_names = yamnet_model.class_names('yamnet/yamnet_class_map.csv')
        params.PATCH_HOP_SECONDS = 0.1  # 10 Hz scores frame rate.
        self.graph = tf.Graph()
        params.SAMPLE_RATE = sr
        with self.graph.as_default():
            self.yamnet = yamnet_model.yamnet_frames_model(params)
            self.yamnet.load_weights('yamnet/yamnet.h5')
            
    def predict_music(self, wav_file_name):
        wav_data, sr = load(wav_file_name)
        with self.graph.as_default():
            scores, spectrogram = self.yamnet.predict(np.reshape(wav_data, [1, -1]), steps=1)
        return wav_data, scores, spectrogram