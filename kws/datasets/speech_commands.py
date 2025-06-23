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
from pathlib import Path
from typing import Dict, List

from loguru import logger

from kws.common.types import LabelIndex
from kws.common.errors import DatasetError, handle_error


@dataclass
class DatasetConfig:
    """Configuration for the speech commands dataset."""

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
    data_url: str = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    random_seed: int = 59185
    download_data: bool = True


class SpeechCommandDataset:
    """Dataset for speech commands classification.

    This dataset class handles downloading, extracting, and organizing
    the Google Speech Commands dataset for keyword spotting tasks.
    """

    SILENCE_LABEL = "_silence_"
    UNKNOWN_LABEL = "_unknown_"
    BACKGROUND_NOISE_DIR = "_background_noise_"
    MAX_NUM_WAVS_PER_CLASS = 2**27 - 1

    def __init__(self, config: DatasetConfig, root_dir: Path):
        """
        Initialize the dataset.

        Args:
            config: Dataset configuration
            root_dir: Root directory for dataset

        Raises:
            DatasetError: If there's an error downloading or preparing the dataset
        """
        self.config = config
        self.root_dir = root_dir
        self.data_index = {"training": [], "validation": [], "testing": []}
        self.word_to_index = {}

        # Initialize dataset
        try:
            if self.config.download_data:
                self._download_and_extract()

            self.words_list = self.prepare_word_list(self.config.wanted_words)
            self._prepare_data_index()
        except Exception as e:
            handle_error(e, DatasetError, "Failed to initialize dataset")

    def _download_and_extract(self) -> None:
        """Download and extract the dataset if not already present.

        Raises:
            DatasetError: If there's an error downloading or extracting the dataset
        """
        filename = self.config.data_url.split("/")[-1]
        filepath = self.root_dir / filename

        if filepath.exists():
            logger.info(f"Dataset archive already exists at {filepath}")
            return

        self.root_dir.mkdir(parents=True, exist_ok=True)

        def progress_hook(count, block_size, total_size):
            percent = float(count * block_size) / float(total_size) * 100.0
            sys.stdout.write(f"\r>> Downloading {filename} {percent:.1f}%")
            sys.stdout.flush()

        try:
            logger.info(f"Downloading dataset from {self.config.data_url}")
            filepath, _ = urllib.request.urlretrieve(
                self.config.data_url, filepath, progress_hook
            )

            logger.info(
                f"Successfully downloaded {filename} ({os.stat(filepath).st_size} bytes)"
            )

            logger.info(f"Extracting dataset to {self.root_dir}")
            with tarfile.open(filepath, "r:gz") as tar:
                tar.extractall(self.root_dir)

            logger.info("Dataset extraction completed")
        except Exception as e:
            raise DatasetError(f"Failed to download or extract dataset: {e}") from e

    def prepare_word_list(self, wanted_words: List[str]) -> List[str]:
        """Prepare word list by adding special labels.

        Args:
            wanted_words: List of wanted word classes

        Returns:
            Complete list of words including silence and unknown labels
        """
        return [self.SILENCE_LABEL, self.UNKNOWN_LABEL] + wanted_words

    def _prepare_data_index(self) -> None:
        """Prepare data index for training, validation, and testing.

        Organizes the dataset files into training, validation and testing splits
        while handling special cases like silence and unknown words.

        Raises:
            DatasetError: If there's an error preparing the data index
        """
        random.seed(self.config.random_seed)

        try:
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
                raise DatasetError("No words found in dataset")

            for wanted_word in self.config.wanted_words:
                if wanted_word not in all_words:
                    raise DatasetError(
                        f"Expected word {wanted_word} not found in dataset"
                    )
        except Exception as e:
            if isinstance(e, DatasetError):
                raise
            raise DatasetError(f"Error preparing data index: {str(e)}") from e

            # Process silence samples
            try:
                silence_wav_path = self.data_index["training"][0]["file"]
            except (IndexError, KeyError) as e:
                raise DatasetError(
                    "No training samples available for silence reference"
                ) from e

            for set_index in ["validation", "testing", "training"]:
                set_size = len(self.data_index[set_index])

                # Add silence samples
                silence_size = int(
                    math.ceil(set_size * self.config.silence_percentage / 100)
                )
                for _ in range(silence_size):
                    self.data_index[set_index].append(
                        {"label": self.SILENCE_LABEL, "file": silence_wav_path}
                    )

                # Add unknown word samples
                random.shuffle(unknown_index[set_index])
                unknown_size = int(
                    math.ceil(set_size * self.config.unknown_percentage / 100)
                )
                self.data_index[set_index].extend(
                    unknown_index[set_index][:unknown_size]
                )

            # Make sure the ordering is random
            for split in ["validation", "testing", "training"]:
                random.shuffle(self.data_index[split])

            # Create word to index mapping
            self.words_list = self.prepare_word_list(self.config.wanted_words)
            self.word_to_index = {}
            for word in all_words:
                if word in wanted_words_index:
                    self.word_to_index[word] = wanted_words_index[word]
                else:
                    self.word_to_index[word] = LabelIndex.UNKNOWN_WORD_INDEX
            self.word_to_index[self.SILENCE_LABEL] = LabelIndex.SILENCE_INDEX

    def get_data(self, split: str) -> List[Dict]:
        """Get data for a specific split.

        Args:
            split: One of 'training', 'validation', or 'testing'

        Returns:
            List of dicts with 'file' and 'label' keys

        Raises:
            DatasetError: If the split is invalid
        """
        if split not in self.data_index:
            raise DatasetError(
                f"Invalid split: {split}. Must be one of {list(self.data_index.keys())}"
            )
        return self.data_index[split]

    def __len__(self) -> int:
        """Get total number of samples."""
        return sum(len(data) for data in self.data_index.values())

    def get_class_count(self) -> int:
        """Get the number of classes (words)."""
        return len(self.words_list)

    def get_words_list(self) -> List[str]:
        """Get the list of words (classes)."""
        return self.words_list

    def which_set(self, filename: str) -> str:
        """Determine which set (training, validation, testing) a file belongs to.

        This uses a hash of the filename to ensure consistent splitting across runs.

        Args:
            filename: The audio file path

        Returns:
            One of 'training', 'validation', or 'testing'
        """
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
