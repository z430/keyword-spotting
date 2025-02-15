import glob
import hashlib
import math
import os
import random
import re
import sys
import tarfile
import urllib.request
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Dict, List

from loguru import logger


class LabelIndex(IntEnum):
    SILENCE_INDEX = 0
    UNKNOWN_WORD_INDEX = 1


@dataclass
class DatasetConfig:
    silence_percentage: float = 10
    unknown_percentage: float = 10
    testing_percentage: float = 10
    validation_percentage: float = 10
    wanted_words: List[str] = field(
        default_factory=lambda: [
            "yes",
            "no",
            "up",
            "down",
            "left",
            "right",
            "on",
            "off",
            "stop",
            "go",
        ]
    )


class SpeechCommandDataset:
    """Dataset for speech commands classification."""

    SILENCE_LABEL = "_silence_"
    UNKNOWN_LABEL = "_unknown_"
    BACKGROUND_NOISE_DIR = "_background_noise_"
    DATA_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    MAX_NUM_WAVS_PER_CLASS = 2**27 - 1
    RANDOM_SEED = 59185

    def __init__(self, config: DatasetConfig, root_dir: Path):
        """
        Initialize the dataset.

        Args:
            config: Dataset configuration
            root_dir: Root directory for dataset
        """
        self.config = config
        self.root_dir = root_dir

        # Initialize dataset
        self._download_and_extract()
        self.words_list = self.prepare_word_list(self.config.wanted_words)
        self._prepare_data_index()

    def _download_and_extract(self) -> None:
        """Download and extract the dataset if not already present."""
        filename = self.DATA_URL.split("/")[-1]
        filepath = self.root_dir / filename

        if filepath.exists():
            return

        self.root_dir.mkdir(parents=True, exist_ok=True)

        def progress_hook(count, block_size, total_size):
            percent = float(count * block_size) / float(total_size) * 100.0
            sys.stdout.write(f"\r>> Downloading {filename} {percent:.1f}%")
            sys.stdout.flush()

        try:
            filepath, _ = urllib.request.urlretrieve(
                self.DATA_URL, filepath, progress_hook
            )
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise

        logger.info(
            f"Successfully downloaded {filename} ({os.stat(filepath).st_size} bytes)"
        )
        tarfile.open(filepath, "r:gz").extractall(self.root_dir)

    def prepare_word_list(self, wanted_words: List[str]) -> List[str]:
        return [self.SILENCE_LABEL, self.UNKNOWN_LABEL] + wanted_words

    def _prepare_data_index(self):
        """Prepare data index for training, validation, and testing."""
        random.seed(self.RANDOM_SEED)

        wanted_words_index = {}
        for index, wanted_word in enumerate(self.config.wanted_words):
            wanted_words_index[wanted_word] = index + 2

        self.data_index = {"training": [], "validation": [], "testing": []}
        unknown_index = {"training": [], "validation": [], "testing": []}

        all_words = {}

        # Collect all audio files
        search_path = os.path.join(str(self.root_dir), "*", "*.wav")
        for wav_path in glob.glob(search_path):
            _, word = os.path.split(os.path.dirname(wav_path))
            word = word.lower()
            if word == self.BACKGROUND_NOISE_DIR:
                continue

            all_words[word] = True
            set_index = self.which_set(wav_path)

            if word in self.config.wanted_words:
                self.data_index[set_index].append({"label": word, "file": wav_path})
            else:
                unknown_index[set_index].append({"label": word, "file": wav_path})

        if not all_words:
            raise ValueError("No words found in dataset")

        for index, wanted_word in enumerate(self.config.wanted_words):
            if wanted_word not in all_words:
                raise ValueError(f"Expected word {wanted_word} not found in dataset")

        silence_wav_path = self.data_index["training"][0]["file"]
        for set_index in ["validation", "testing", "training"]:
            set_size = len(self.data_index[set_index])

            silence_size = int(
                math.ceil(set_size * self.config.silence_percentage / 100)
            )
            for _ in range(silence_size):
                self.data_index[set_index].append(
                    {"label": self.SILENCE_LABEL, "file": silence_wav_path}
                )

            random.shuffle(unknown_index[set_index])
            unknown_size = int(
                math.ceil(set_size * self.config.unknown_percentage / 100)
            )
            self.data_index[set_index].extend(unknown_index[set_index][:unknown_size])

        # make sure the ordering is random
        for set_index in ["validation", "testing", "training"]:
            random.shuffle(self.data_index[set_index])

        self.words_list = self.prepare_word_list(self.config.wanted_words)
        self.word_to_index = {}
        for word in all_words:
            if word in wanted_words_index:
                self.word_to_index[word] = wanted_words_index[word]
            else:
                self.word_to_index[word] = LabelIndex.UNKNOWN_WORD_INDEX
        self.word_to_index[self.SILENCE_LABEL] = LabelIndex.SILENCE_INDEX

    def get_data(self, split: str) -> List[Dict]:
        """Get data for a specific split."""
        return self.data_index[split]

    def __len__(self) -> int:
        """Get total number of samples."""
        return sum(len(data) for data in self.data_index.values())

    def which_set(self, filename: str) -> str:
        base_name = os.path.basename(filename)
        hash_name = re.sub(r"_nohash_.*$", "", base_name)
        hash_name_hashed = hashlib.sha1(hash_name.encode()).hexdigest()
        percentage_hash = (
            int(hash_name_hashed, 16) % (self.MAX_NUM_WAVS_PER_CLASS + 1)
        ) * (100.0 / self.MAX_NUM_WAVS_PER_CLASS)
        if percentage_hash < self.config.validation_percentage:
            return "validation"
        if percentage_hash < (
            self.config.testing_percentage + self.config.validation_percentage
        ):
            return "testing"
        return "training"
