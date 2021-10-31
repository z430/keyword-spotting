from typing import Tuple

import pandas as pd
import tensorflow as tf
from tensorflow._api.v2 import audio, data


class SpeechCommandLoader:
    def __init__(
        self, sample_rate, audio_length, batch_size, autotune, feature="spectrogram"
    ) -> None:
        self.sample_rate = sample_rate
        self.sample_size = audio_length
        self.feature = feature
        self.autotune = autotune
        self.batch_size = batch_size
        self.shuffle = True

    def to_tf_dataset(self, dataset: pd.DataFrame) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_tensor_slices((dataset.file, dataset.label))
        dataset = dataset.map(self.read_audiofile, num_parallel_calls=self.autotune)
        dataset = dataset.map(self.spectrogram, num_parallel_calls=self.autotune)
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.cache().prefetch(self.autotune)
        return dataset

    @staticmethod
    def get_input_shape(dataset):
        for feature, label in dataset.take(1):
            input_shape = feature.shape[1:]
        return input_shape

    def decode_audio(self, audio_name: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Decode audio from tf
        The audio samples already normalized [-1, 1]

        Args:
            audio_name (tf.Tensor): path of audio

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: the audio samples (float32)
                                         the sample rate (int64)
        """
        audio_binary = tf.io.read_file(audio_name)
        waveform, sample_rate = tf.audio.decode_wav(
            audio_binary, desired_samples=self.sample_size
        )
        waveform = tf.squeeze(waveform, axis=-1)
        return waveform, sample_rate

    def read_audiofile(
        self, audio_name: tf.Tensor, audio_label: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Read audio and load the audio file for training

        Args:
            audio_name (tf.Tensor): path of audio
            audio_label (tf.Tensor): the label of the audio

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: transformed audio with fixed length (float32)
                                         label (int64)
        """
        # read audio with tf
        waveform, sr = self.decode_audio(audio_name=audio_name)
        return waveform, audio_label

    def spectrogram(
        self, audio: tf.Tensor, label: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """get spectrogram feature

        Args:
            audio (tf.Tensor): waveform
            label (tf.Tensor): label

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: spectrogram (float64), label (int64)
        """
        audio = tf.cast(audio, tf.float32)
        spectrogram = tf.signal.stft(audio, frame_length=255, frame_step=128)
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.expand_dims(spectrogram, axis=-1)
        return spectrogram, label
