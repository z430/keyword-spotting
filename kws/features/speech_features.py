import tensorflow as tf
import python_speech_features as psf
import numpy as np
from pycochleagram import cochleagram as cgram
import librosa


class SpeechFeatures:
    def __init__(self):
        self.fs = 16000

        # mfcc 49x40
        self.win_length = 0.04
        self.hop_length = 0.02
        self.numcep = 40
        self.nfilt = 40
        self.n_fft = 1024

    def mfcc(self, sig):
        # print(self.win_length, self.hop_length)
        mfcc = psf.mfcc(
            sig,
            winlen=self.win_length,
            nfft=self.n_fft,
            winstep=self.hop_length,
            numcep=self.numcep,
            nfilt=self.nfilt,
        )
        return mfcc

    def get_mfcc(self, waveform, label):
        waveform = tf.cast(waveform, tf.float32)
        mfcc_features = tf.py_function(self.mfcc, [waveform.numpy(), tf.float32])
        mfcc_features = tf.convert_to_tensor(mfcc_features)
        return mfcc_features, label

    def cgram_(self, sig):
        cg = cgram.human_cochleagram(
            sig,
            self.fs,
            n=20,
            sample_factor=2,
            downsample=20,
            nonlinearity="power",
            strict=False,
        )
        return cg
