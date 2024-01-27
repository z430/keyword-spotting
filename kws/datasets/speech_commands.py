import glob
import hashlib
import math
import os
import os.path
import random
import re
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
from loguru import logger
from pydantic import BaseModel

from kws.libs.signal_processings import preempashis
from kws.utils.loader import load_yaml

DATA_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134MB
RANDOM_SEED = 59185
SILENCE_INDEX = 0
SILENCE_LABEL = "_silence_"
UNKNOWN_WORD_INDEX = 1
UNKNOWN_WORD_LABEL = "_unknown_"
BACKGROUND_NOISE_DIR_NAME = "_background_noise_"
DEFAULT_DATASET_PATH = Path.home()


class Parameters(BaseModel):
    background_volume: float
    background_frequency: float
    time_shift_ms: float
    sample_rate: int
    clip_duration_ms: int
    use_background_noise: bool

    silence_percentage: float
    unknown_percentage: float
    testing_percentage: float
    validation_percentage: float

    wanted_words: List[str]

    @property
    def time_shift(self) -> int:
        return int((self.time_shift_ms * self.sample_rate) / 1000)

    @property
    def desired_samples(self) -> int:
        return int(self.sample_rate * (self.clip_duration_ms) / 1000)


class SpeechCommandsDataset:
    def __init__(self, parameters_path: Path, dataset_path: Path):
        self.parameters = Parameters(**load_yaml(parameters_path))
        self.dataset_path = dataset_path

        self.maybe_download_and_extract_dataset()
        self.words_list = ["_silence_", "_unknown_"] + self.parameters.wanted_words
        self.prepare_data_index()

    def maybe_download_and_extract_dataset(self):
        filename = DATA_URL.split("/")[-1]
        filepath = self.dataset_path / filename

        self.dataset_path.mkdir(parents=True, exist_ok=True)
        if not filepath.exists():

            def _progress(count, block_size, total_size):
                sys.stdout.write(
                    "\r>> Downloading %s %.1f%%"
                    % (filename, float(count * block_size) / float(total_size) * 100.0)
                )
                sys.stdout.flush()

            try:
                filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
            except:
                logger.error(
                    "Failed to download URL: %s to folder: %s", DATA_URL, filepath
                )
                logger.error(
                    "Please make sure you have enough free space and"
                    " an internet connection"
                )
                raise

            statinfo = os.stat(filepath)
            logger.info(
                "Successfully downloaded %s (%d bytes)", filename, statinfo.st_size
            )
            tarfile.open(filepath, "r:gz").extractall(self.dataset_path)

    def which_set(self, filename, validation_percentage, testing_percentage):
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
        percentage_hash = (int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)) * (
            100.0 / MAX_NUM_WAVS_PER_CLASS
        )
        if percentage_hash < validation_percentage:
            result = "validation"
        elif percentage_hash < (testing_percentage + validation_percentage):
            result = "testing"
        else:
            result = "training"
        return result

    def prepare_data_index(self) -> None:
        random.seed(RANDOM_SEED)

        wanted_words_index = {}
        for index, wanted_word in enumerate(self.parameters.wanted_words):
            wanted_words_index[wanted_word] = index + 2

        self.data_index = {"validation": [], "testing": [], "training": []}
        unknown_index = {"validation": [], "testing": [], "training": []}

        all_words = {}

        search_path = os.path.join(str(self.dataset_path), "*", "*.wav")
        self.get_audio_path(search_path, all_words, wanted_words_index, unknown_index)
        self.check_audio_path(all_words, self.parameters.wanted_words, search_path)
        self.set_silence_data(unknown_index)
        self.set_data_index(all_words, wanted_words_index)

    def set_data_index(self, all_words: Dict, wanted_words_index: Dict) -> None:
        # Make sure the ordering is random.
        for set_index in ["validation", "testing", "training"]:
            random.shuffle(self.data_index[set_index])
        # Prepare the rest of the result data structure.
        self.word_to_index = {}
        for word in all_words:
            if word in wanted_words_index:
                self.word_to_index[word] = wanted_words_index[word]
            else:
                self.word_to_index[word] = UNKNOWN_WORD_INDEX
        self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX

    def set_silence_data(self, unknown_index: Dict) -> None:
        # We need an arbitrary file to load as the input for the silence samples.
        # It's multiplied by zero later, so the content doesn't matter.
        silence_wav_path = self.data_index["training"][0]["file"]
        for set_index in ["validation", "testing", "training"]:
            set_size = len(self.data_index[set_index])
            silence_size = int(
                math.ceil(set_size * self.parameters.silence_percentage / 100)
            )
            for _ in range(silence_size):
                self.data_index[set_index].append(
                    {"label": SILENCE_LABEL, "file": silence_wav_path}
                )
            # Pick some unknowns to add to each partition of the data set.
            random.shuffle(unknown_index[set_index])
            unknown_size = int(
                math.ceil(set_size * self.parameters.unknown_percentage / 100)
            )
            self.data_index[set_index].extend(unknown_index[set_index][:unknown_size])

    def check_audio_path(
        self, all_words: Dict, wanted_words: List[str], search_path: str
    ) -> None:
        if not all_words:
            raise Exception("No .wavs found at " + search_path)
        for _, wanted_word in enumerate(wanted_words):
            if wanted_word not in all_words:
                raise Exception(
                    "Expected to find "
                    + wanted_word
                    + " in labels but only found "
                    + ", ".join(all_words.keys())
                )

    def get_audio_path(
        self, path: str, all_words: Dict, wanted_words_index: Dict, unknown_index: Dict
    ) -> None:
        for wav_path in glob.glob(path):
            _, word = os.path.split(os.path.dirname(wav_path))
            word = word.lower()
            # Treat the '_background_noise_' folder as a special case, since we expect
            # it to contain long audio samples we mix in to improve training.
            if word == BACKGROUND_NOISE_DIR_NAME:
                continue
            all_words[word] = True
            set_index = self.which_set(
                wav_path,
                self.parameters.validation_percentage,
                self.parameters.testing_percentage,
            )
            # If it's a known class, store its detail, otherwise add it to the list
            # we'll use to train the unknown label.
            if word in wanted_words_index:
                self.data_index[set_index].append({"label": word, "file": wav_path})
            else:
                unknown_index[set_index].append({"label": word, "file": wav_path})

    def set_size(self, mode):
        return len(self.data_index[mode])

    def prepare_background_data(self):
        self.background_data = []

        self.dataset_path: Path
        background_dir = self.dataset_path / BACKGROUND_NOISE_DIR_NAME

        if not background_dir.exists():
            return self.background_data

        search_path = os.path.join(
            str(self.dataset_path), BACKGROUND_NOISE_DIR_NAME, "*.wav"
        )
        for wav_path in glob.glob(search_path):
            wav_data, _ = librosa.load(wav_path, sr=self.parameters.sample_rate)
            self.background_data.append(wav_data)

        if not self.background_data:
            raise Exception("No background wav files were found in " + search_path)

    def get_datafiles(self, mode):
        return self.data_index[mode]

    def audio_transform(self, filename, label):
        """audio_transform.

        :param filename: tf string audio path
        :param label: the label of audio
        """
        # print(filename, label)
        # read audio with librosa
        audio, sample_rate = librosa.load(
            filename.numpy().decode("UTF-8"), sr=self.parameters.sample_rate
        )
        # fix the audio length
        audio = librosa.util.fix_length(audio, size=self.parameters.desired_samples)
        # preemphasis -> make the audio gain higher
        audio = preempashis(audio)

        # if the label is silence make the audio volume to 0
        if label.numpy() == SILENCE_INDEX:
            audio = audio * 0

        # audio augmentation
        # 1. time shifting
        time_shift_amount = np.random.randint(
            -self.parameters.time_shift, self.parameters.time_shift
        )
        if time_shift_amount > 0:
            time_shift_padding = [time_shift_amount, 0]
            time_shift_offset = 0
        else:
            time_shift_padding = [0, -time_shift_amount]
            time_shift_offset = -time_shift_amount

        padded_foreground = np.pad(audio, time_shift_padding, "constant")

        sliced_foreground = librosa.util.fix_length(
            padded_foreground[time_shift_offset:], size=self.parameters.desired_samples
        )

        # 2. select noise type and randomly select how big the volume is
        if self.parameters.use_background_noise or label.numpy() == SILENCE_INDEX:
            background_index = np.random.randint(len(self.background_data))
            background_samples = self.background_data[background_index]
            background_offset = np.random.randint(
                0, len(background_samples) - self.parameters.desired_samples
            )
            background_clipped = background_samples[
                background_offset : (
                    background_offset + self.parameters.desired_samples
                )
            ]
            background_reshaped = background_clipped.reshape(
                self.parameters.desired_samples
            )
            if label.numpy() == SILENCE_INDEX:
                background_volume = np.random.uniform(0, 1)
            elif np.random.uniform(0, 1) < self.parameters.background_frequency:
                background_volume = np.random.uniform(
                    0, self.parameters.background_volume
                )
            else:
                background_volume = 0
        else:
            background_reshaped = np.zeros(self.parameters.desired_samples)
            background_volume = 0

        # adjust the noisy signal volume
        background_mul = np.multiply(background_reshaped, background_volume)
        # mix audio with noisy signal
        audio = np.add(background_mul, sliced_foreground)
        # print(audio.shape)
        return audio
