""" Training """

from logging import BufferingFormatter
import random
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models

import pandas as pd
import wandb
from wandb.keras import WandbCallback

from speech_features import SpeechFeatures
import input_data

wanted_words = "left,right,forward,backward,stop,go"
features = input_data.GetData(wanted_words=wanted_words, feature="mfcc")
AUTOTUNE = tf.data.experimental.AUTOTUNE


def rosa_read(filename, label):
    waveform = tf.py_function(features.audio_transform, [filename, label], [tf.float32])
    waveform = tf.convert_to_tensor(waveform)
    waveform = tf.squeeze(waveform, axis=0)
    return waveform, label


def get_spectrogram(waveform, label):
    waveform = tf.cast(waveform, tf.float32)
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    return spectrogram, label


def preprocess_dataset(dataset):
    files_ds = tf.data.Dataset.from_tensor_slices((dataset["file"], dataset["label"]))
    output_ds = files_ds.map(rosa_read, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(get_spectrogram, num_parallel_calls=AUTOTUNE)
    return output_ds


def main():
    """------------------- Features Configuration -------------------"""
    wandb.init(
        project="tf-keywork-spotting",
        config={
            "learning_rate": 0.005,
            "batch_size": 16,
            "epochs": 100,
            "loss_function": "sparse_categorical_crossentropy",
            "architecture": "simpleCNN",
            "wanted_words": wanted_words.split(","),
        },
    )

    config = wandb.config

    # initialize keras
    tf.keras.backend.clear_session()

    training_files = features.get_datafiles("training")
    validation_files = features.get_datafiles("validation")

    # transform the list dicts into dataframe
    training_data = pd.DataFrame(training_files)
    training_data["label"] = [
        features.word_to_index[label] for label in training_data["label"]
    ]

    validation_data = pd.DataFrame(validation_files)
    validation_data["label"] = [
        features.word_to_index[label] for label in validation_data["label"]
    ]

    training_ds = preprocess_dataset(training_data)
    validation_ds = preprocess_dataset(validation_data)

    training_ds = training_ds.shuffle(buffer_size=100)
    validation_ds = validation_ds.shuffle(buffer_size=100)

    training_ds = training_ds.batch(config.batch_size)
    validation_ds = validation_ds.batch(config.batch_size)

    training_ds = training_ds.cache().prefetch(AUTOTUNE)
    validation_ds = validation_ds.cache().prefetch(AUTOTUNE)

    for spectrogram, labels in training_ds.take(1):
        input_shape = spectrogram.shape[1:]
        print(input_shape, labels)

    num_labels = len(features.words_list)
    print(f"Input Shape: {input_shape}, len labels: {num_labels}")

    norm_layer = preprocessing.Normalization()
    norm_layer.adapt(training_ds.map(lambda x, _: x))

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            preprocessing.Resizing(32, 32),
            norm_layer,
            layers.Conv2D(32, 3, activation="relu"),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_labels),
        ]
    )

    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    history = model.fit(
        training_ds,
        validation_data=validation_ds,
        epochs=config.epochs,
        callbacks=[WandbCallback()],
    )


if __name__ == "__main__":
    main()
