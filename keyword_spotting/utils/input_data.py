""" This class is for pick the train, val and test set
    All the the data augmentation and data setting are defined
    in here
"""

import glob
import hashlib
import math
import os
import os.path
import random
import re
import sys
import tarfile
from typing import Dict
import urllib

import librosa
import pandas as pd
import python_speech_features as psf
import numpy as np
import tensorflow as tf
import tqdm


class GetData:
    def __init__(self, prepare_data=True, wanted_words="marvin"):
        # don't change this parameter
        self.prepare_data = prepare_data
        self.MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134MB
        self.RANDOM_SEED = 59185
        self.SILENCE_INDEX = 0
        self.SILENCE_LABEL = "_silence_"
        self.UNKNOWN_WORD_INDEX = 1
        self.UNKNOWN_WORD_LABEL = "_unknown_"
        self.BACKGROUND_NOISE_DIR_NAME = "_background_noise_"

        self.data_url = (
            "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
        )
        self.data_dir = "../data/train"
        if os.path.isdir(self.data_dir):
            self.maybe_download_and_extract_dataset(self.data_url, self.data_dir)
        else:
            os.makedirs(self.data_dir)
            self.maybe_download_and_extract_dataset(self.data_url, self.data_dir)

        # audio augmentation params
        self.background_volume = 0.1
        self.background_frequency = 0.8
        self.time_shift_ms = 50.0
        self.sample_rate = 16000
        self.clip_duration_ms = 1000
        self.time_shift = int((self.time_shift_ms * self.sample_rate) / 1000)
        self.desired_samples = int(self.sample_rate * (self.clip_duration_ms / 1000))
        self.use_background_noise = True

        self.silence_percentage = 10.0
        self.unknown_percentage = 10.0
        self.testing_percentage = 10.0
        self.validation_percentage = 10.0
        self.wanted_words = wanted_words

        # initialization
        self.words_list = self.prepare_word_list(self.wanted_words)
        self.prepare_data_index(
            self.silence_percentage,
            self.unknown_percentage,
            self.wanted_words,
            self.validation_percentage,
            self.testing_percentage,
        )
        self.prepare_background_data()
        for k, v in self.data_index.items():
            setattr(self, k, v)

    def transform_df(self, dataset: Dict) -> pd.DataFrame:
        dataset = pd.DataFrame(dataset)
        dataset.label = [self.word_to_index[label] for label in dataset.label]
        return dataset

    @staticmethod
    def prepare_word_list(wanted_words):
        return ["_silence_", "_unknown_"] + wanted_words

    def maybe_download_and_extract_dataset(self, data_url, dest_directory):
        """
        Download and extract data set tar file.
        If the data set we're using doesn't already exist, this function
        downloads it from the TensorFlow.org website and unpacks it into a
        directory.
        If the data_url is none, don't download anything and expect the data
        directory to contain the correct files already.
        Args:
        data_url: Web location of the tar file containing the data set.
        dest_directory: File path to extract data to.
        """
        if not data_url:
            return
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = data_url.split("/")[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):

            def _progress(count, block_size, total_size):
                sys.stdout.write(
                    "\r>> Downloading %s %.1f%%"
                    % (filename, float(count * block_size) / float(total_size) * 100.0)
                )
                sys.stdout.flush()

            try:
                filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
            except:
                print(f"Failed to download URL: {data_url} to folder: {filepath}")
                print(
                    "Please make sure you have enough free space and"
                    " an internet connection"
                )
                raise
            print()
            statinfo = os.stat(filepath)
            print(f"Successfully downloaded {filename} ({statinfo.st_size} bytes)")
            tarfile.open(filepath, "r:gz").extractall(dest_directory)

    def which_set(self, filename, validation_percentage, testing_percentage):
        """
        Determines which data partition the file should belong to.

        We want to keep files in the same training, validation, or testing sets even
        if new ones are added over time. This makes it less likely that testing
        samples will accidentally be reused in training when long runs are restarted
        for example. To keep this stability, a hash of the filename is taken and used
        to determine which set it should belong to. This determination only depends on
        the name and the set proportions, so it won't change as other files are added.

        It's also useful to associate particular files as related (for example words
        spoken by the same person), so anything after '_nohash_' in a filename is
        ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
        'bobby_nohash_1.wav' are always in the same set, for example.

        Args:
         filename: File path of the data sample.
         validation_percentage: How much of the data set to use for validation.
         testing_percentage: How much of the data set to use for testing.

        Returns:
         String, one of 'training', 'validation', or 'testing'.
        """
        base_name = os.path.basename(filename)
        # We want to ignore anything after '_nohash_' in the file name when
        # deciding which set to put a wav in, so the data set creator has a way of
        # grouping wavs that are close variations of each other.
        hash_name = re.sub(r"_nohash_.*$", "", base_name)
        # print(type(hash_name))
        # This looks a bit magical, but we need to decide whether this file should
        # go into the training, testing, or validation sets, and we want to keep
        # existing files in the same set even if more files are subsequently
        # added.
        # To do that, we need a stable way of deciding based on just the file name
        # itself, so we do a hash of that and then use that to generate a
        # probability value that we use to assign it.
        # hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
        hash_name_hashed = hashlib.sha1(hash_name.encode()).hexdigest()
        percentage_hash = (
            int(hash_name_hashed, 16) % (self.MAX_NUM_WAVS_PER_CLASS + 1)
        ) * (100.0 / self.MAX_NUM_WAVS_PER_CLASS)
        if percentage_hash < validation_percentage:
            result = "validation"
        elif percentage_hash < (testing_percentage + validation_percentage):
            result = "testing"
        else:
            result = "training"
        return result

    def prepare_data_index(
        self,
        silence_percentage,
        unknown_percentage,
        wanted_words,
        validation_percentage,
        testing_percentage,
    ):
        """
        Prepares a list of the samples organized by set and label.
        The training loop needs a list of all the available data, organized by
        which partition it should belong to, and with ground truth labels attached.
        This function analyzes the folders below the `data_dir`, figures out the
        right
        labels for each file based on the name of the subdirectory it belongs to,
        and uses a stable hash to assign it to a data set partition.
        Args:
          silence_percentage: How much of the resulting data should be background.
          unknown_percentage: How much should be audio outside the wanted classes.
          wanted_words: Labels of the classes we want to be able to recognize.
          validation_percentage: How much of the data set to use for validation.
          testing_percentage: How much of the data set to use for testing.
        Returns:
          Dictionary containing a list of file information for each set partition,
          and a lookup map for each class to determine its numeric index.
        Raises:
          Exception: If expected files are not found.
        """
        # Make sure the shuffling and picking of unknowns is deterministic.
        random.seed(self.RANDOM_SEED)
        wanted_words_index = {}
        for index, wanted_word in enumerate(wanted_words):
            wanted_words_index[wanted_word] = index + 2
        self.data_index = {"validation": [], "testing": [], "training": []}
        unknown_index = {"validation": [], "testing": [], "training": []}
        all_words = {}
        # Look through all the subfolders to find audio samples
        search_path = os.path.join(self.data_dir, "*", "*.wav")
        for wav_path in glob.glob(search_path):
            _, word = os.path.split(os.path.dirname(wav_path))
            word = word.lower()
            # Treat the '_background_noise_' folder as a special case, since we expect
            # it to contain long audio samples we mix in to improve training.
            if word == self.BACKGROUND_NOISE_DIR_NAME:
                continue
            all_words[word] = True
            set_index = self.which_set(
                wav_path, validation_percentage, testing_percentage
            )
            # If it's a known class, store its detail, otherwise add it to the list
            # we'll use to train the unknown label.
            if word in wanted_words_index:
                self.data_index[set_index].append({"label": word, "file": wav_path})
            else:
                unknown_index[set_index].append({"label": word, "file": wav_path})
        if not all_words:
            raise Exception("No .wavs found at " + search_path)
        for index, wanted_word in enumerate(wanted_words):
            if wanted_word not in all_words:
                raise Exception(
                    "Expected to find "
                    + wanted_word
                    + " in labels but only found "
                    + ", ".join(all_words.keys())
                )
        # We need an arbitrary file to load as the input for the silence samples.
        # It's multiplied by zero later, so the content doesn't matter.
        silence_wav_path = self.data_index["training"][0]["file"]
        for set_index in ["validation", "testing", "training"]:
            set_size = len(self.data_index[set_index])
            silence_size = int(math.ceil(set_size * silence_percentage / 100))
            for _ in range(silence_size):
                self.data_index[set_index].append(
                    {"label": self.SILENCE_LABEL, "file": silence_wav_path}
                )
            # Pick some unknowns to add to each partition of the data set.
            random.shuffle(unknown_index[set_index])
            unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
            self.data_index[set_index].extend(unknown_index[set_index][:unknown_size])
        # Make sure the ordering is random.
        for set_index in ["validation", "testing", "training"]:
            random.shuffle(self.data_index[set_index])
        # Prepare the rest of the result data structure.
        self.words_list = self.prepare_word_list(wanted_words)
        self.word_to_index = {}
        for word in all_words:
            if word in wanted_words_index:
                self.word_to_index[word] = wanted_words_index[word]
            else:
                self.word_to_index[word] = self.UNKNOWN_WORD_INDEX
        self.word_to_index[self.SILENCE_LABEL] = self.SILENCE_INDEX

    def set_size(self, mode):
        """
        Calculates the number of samples in the dataset partition.
        Args:
          mode: Which partition, must be 'training', 'validation', or 'testing'.
        Returns:
          Number of samples in the partition.
        """
        return len(self.data_index[mode])

    def prepare_background_data(self):
        """Searches a folder for background noise audio, and loads it into memory.
        It's expected that the background audio samples will be in a subdirectory
        named '_background_noise_' inside the 'data_dir' folder, as .wavs that match
        the sample rate of the training data, but can be much longer in duration.
        If the '_background_noise_' folder doesn't exist at all, this isn't an
        error, it's just taken to mean that no background noise augmentation should
        be used. If the folder does exist, but it's empty, that's treated as an
        error.
        Returns:
          List of raw PCM-encoded audio samples of background noise.
        Raises:
          Exception: If files aren't found in the folder.
        """
        self.background_data = []
        background_dir = os.path.join(self.data_dir, self.BACKGROUND_NOISE_DIR_NAME)
        if not os.path.exists(background_dir):
            return self.background_data

        search_path = os.path.join(
            self.data_dir, self.BACKGROUND_NOISE_DIR_NAME, "*.wav"
        )
        for wav_path in glob.glob(search_path):
            wav_data, _ = librosa.load(wav_path, sr=self.sample_rate)
            self.background_data.append(wav_data)
        if not self.background_data:
            raise Exception("No background wav files were found in " + search_path)

    def audio_transform(self, filename: tf.string, label: tf.string) -> tf.float32:
        """Read audio and load audio

        Args:
            filename (tf.string): path of audio
            label (tf.string): the label name of the audio

        Returns:
            tf.float32: transformed audio
        """
        # read audio with librosa
        audio, sample_rate = librosa.load(
            filename.numpy().decode("UTF-8"), sr=self.sample_rate
        )
        # fix the audio length
        audio = librosa.util.fix_length(audio, self.desired_samples)
        # preemphasis -> make the audio gain higher
        audio = psf.sigproc.preemphasis(audio)

        # if the label is silence make the audio volume to 0
        if label.numpy() == self.SILENCE_INDEX:
            audio = audio * 0

        # audio augmentation
        # 1. time shifting
        time_shift_amount = np.random.randint(-self.time_shift, self.time_shift)
        if time_shift_amount > 0:
            time_shift_padding = [time_shift_amount, 0]
            time_shift_offset = 0
        else:
            time_shift_padding = [0, -time_shift_amount]
            time_shift_offset = -time_shift_amount

        padded_foreground = np.pad(audio, time_shift_padding, "constant")
        sliced_foreground = librosa.util.fix_length(
            padded_foreground[time_shift_offset:], self.desired_samples
        )

        # 2. select noise type and randomly select how big the volume is
        if self.use_background_noise or label.numpy() == self.SILENCE_INDEX:
            background_index = np.random.randint(len(self.background_data))
            background_samples = self.background_data[background_index]
            background_offset = np.random.randint(
                0, len(background_samples) - self.desired_samples
            )
            background_clipped = background_samples[
                background_offset : (background_offset + self.desired_samples)
            ]
            background_reshaped = background_clipped.reshape(self.desired_samples)
            if label.numpy() == self.SILENCE_INDEX:
                background_volume = np.random.uniform(0, 1)
            elif np.random.uniform(0, 1) < self.background_frequency:
                background_volume = np.random.uniform(0, self.background_volume)
            else:
                background_volume = 0
        else:
            background_reshaped = np.zeros(self.desired_samples)
            background_volume = 0

        # adjust the noisy signal volume
        background_mul = np.multiply(background_reshaped, background_volume)
        # mix audio with noisy signal
        audio = np.add(background_mul, sliced_foreground)
        # print(audio.shape)
        return audio
