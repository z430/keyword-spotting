import glob
import hashlib
import math
import os
import random
import re
import sys
import tarfile
import urllib.request
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
import python_speech_features as psf
from loguru import logger


class LabelIndex(IntEnum):
    SILENCE_INDEX = 0
    UNKNOWN_WORD_INDEX = 1


class DatasetConfig:
    # Audio parameters
    background_volume: float = 0.1
    background_frequency: float = 0.8
    time_shift_ms: float = 100
    sample_rate: int = 16000
    clip_duration_ms: int = 1000
    use_background_noise: bool = True

    # Dataset split parameters
    silence_percentage: float = 10
    unknown_percentage: float = 10
    testing_percentage: float = 10
    validation_percentage: float = 10
    wanted_words: List[str] = [
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

    @property
    def time_shift(self) -> int:
        return int((self.time_shift_ms * self.sample_rate) / 1000)

    @property
    def desired_samples(self) -> int:
        return int(self.sample_rate * self.clip_duration_ms / 1000)


class AudioProcessor:
    def __init__(self, config: DatasetConfig):
        self.config = config

    def transform(
        self, audio: np.ndarray, label: int, background_data: List[np.ndarray]
    ) -> np.ndarray:
        """Apply audio transformations including time shifting and background noise."""
        # Fix audio length
        audio = librosa.util.fix_length(audio, size=self.config.desired_samples)
        audio = psf.base.sigproc.preemphasis(audio)

        # Handle silence
        if label == LabelIndex.SILENCE_INDEX:
            audio = np.zeros_like(audio)

        # Apply time shifting
        audio = self._apply_time_shift(audio)

        # Add background noise
        if self.config.use_background_noise or label == LabelIndex.SILENCE_INDEX:
            audio = self._add_background_noise(audio, label, background_data)

        return audio

    def _apply_time_shift(self, audio: np.ndarray) -> np.ndarray:
        """Apply random time shifting to audio."""
        time_shift_amount = np.random.randint(
            -self.config.time_shift, self.config.time_shift
        )

        if time_shift_amount > 0:
            padding = [time_shift_amount, 0]
            offset = 0
        else:
            padding = [0, -time_shift_amount]
            offset = -time_shift_amount

        padded_audio = np.pad(audio, padding, "constant")
        return librosa.util.fix_length(
            padded_audio[offset:], size=self.config.desired_samples
        )

    def _add_background_noise(
        self, audio: np.ndarray, label: int, background_data: List[np.ndarray]
    ) -> np.ndarray:
        """Add background noise to audio."""
        if not background_data:
            return audio

        background_samples = random.choice(background_data)
        background_offset = np.random.randint(
            0, len(background_samples) - self.config.desired_samples
        )
        background_clip = background_samples[
            background_offset : background_offset + self.config.desired_samples
        ]

        # Determine background volume
        if label == LabelIndex.SILENCE_INDEX:
            background_volume = np.random.uniform(0, 1)
        elif np.random.uniform(0, 1) < self.config.background_frequency:
            background_volume = np.random.uniform(0, self.config.background_volume)
        else:
            background_volume = 0

        return audio + (background_clip * background_volume)


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
        self.audio_processor = AudioProcessor(config)

        # Initialize dataset
        self._download_and_extract()
        self.words_list = [self.SILENCE_LABEL, self.UNKNOWN_LABEL] + config.wanted_words
        self.word_to_index = self._create_word_to_index()
        self.data_index = self._prepare_data_index()
        self.background_data = self._load_background_data()

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

    def _create_word_to_index(self) -> Dict[str, int]:
        """Create mapping from words to indices."""
        word_to_index = {
            self.SILENCE_LABEL: LabelIndex.SILENCE_INDEX.value,
            self.UNKNOWN_LABEL: LabelIndex.UNKNOWN_WORD_INDEX.value,
        }
        for idx, word in enumerate(self.config.wanted_words):
            word_to_index[word] = idx + 2
        return word_to_index

    def _prepare_data_index(self) -> Dict[str, List[Dict]]:
        """Prepare data index for training, validation, and testing."""
        random.seed(self.RANDOM_SEED)
        data_index = {split: [] for split in ["training", "validation", "testing"]}
        unknown_index = {split: [] for split in ["training", "validation", "testing"]}

        # Collect all audio files
        search_path = os.path.join(str(self.root_dir), "*", "*.wav")
        for wav_path in glob.glob(search_path):
            word = os.path.basename(os.path.dirname(wav_path)).lower()

            if word == self.BACKGROUND_NOISE_DIR:
                continue

            split = self._get_split(wav_path)
            file_data = {"label": word, "file": wav_path}

            if word in self.config.wanted_words:
                data_index[split].append(file_data)
            else:
                unknown_index[split].append(file_data)

        # Add silence and unknown samples
        self._add_silence_and_unknown(data_index, unknown_index)
        return data_index

    def _get_split(self, filename: str) -> str:
        """Determine which split a file belongs to."""
        base_name = os.path.basename(filename)
        hash_name = re.sub(r"_nohash_.*$", "", base_name)
        hash_value = int(hashlib.sha1(hash_name.encode()).hexdigest(), 16)
        percentage_hash = (hash_value % (self.MAX_NUM_WAVS_PER_CLASS + 1)) * (
            100.0 / self.MAX_NUM_WAVS_PER_CLASS
        )

        if percentage_hash < self.config.validation_percentage:
            return "validation"
        elif percentage_hash < (
            self.config.testing_percentage + self.config.validation_percentage
        ):
            return "testing"
        return "training"

    def _add_silence_and_unknown(
        self, data_index: Dict[str, List], unknown_index: Dict[str, List]
    ) -> None:
        """Add silence and unknown samples to each split."""
        silence_wav_path = data_index["training"][0]["file"]

        for split in ["training", "validation", "testing"]:
            set_size = len(data_index[split])

            # Add silence samples
            silence_size = int(
                math.ceil(set_size * self.config.silence_percentage / 100)
            )
            silence_samples = [
                {"label": self.SILENCE_LABEL, "file": silence_wav_path}
                for _ in range(silence_size)
            ]
            data_index[split].extend(silence_samples)

            # Add unknown samples
            unknown_size = int(
                math.ceil(set_size * self.config.unknown_percentage / 100)
            )
            random.shuffle(unknown_index[split])
            data_index[split].extend(unknown_index[split][:unknown_size])

    def _load_background_data(self) -> List[np.ndarray]:
        """Load background noise data."""
        background_path = self.root_dir / self.BACKGROUND_NOISE_DIR
        if not background_path.exists():
            return []

        background_data = []
        for wav_path in background_path.glob("*.wav"):
            audio, _ = librosa.load(wav_path, sr=self.config.sample_rate)
            background_data.append(audio)

        if not background_data:
            logger.warning(f"No background noise files found in {background_path}")

        return background_data

    def get_data(self, split: str) -> List[Dict]:
        """Get data for a specific split."""
        return self.data_index[split]

    def get_audio(self, filename: str, label: str) -> np.ndarray:
        """Load and transform audio file."""
        audio, _ = librosa.load(filename, sr=self.config.sample_rate)
        label_index = self.word_to_index[label]
        return self.audio_processor.transform(audio, label_index, self.background_data)

    def __len__(self) -> int:
        """Get total number of samples."""
        return sum(len(data) for data in self.data_index.values())
