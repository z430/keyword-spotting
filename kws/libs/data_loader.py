"""Data loading utilities for keyword spotting."""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from loguru import logger
from torch.utils.data import Dataset

from kws.common.errors import DatasetError, handle_error
from kws.datasets.speech_commands import SpeechCommandDataset
from kws.libs.audio_processor import AudioProcessor


class SpeechCommandsDataLoader(Dataset):
    """Dataset loader for speech commands.

    This class is responsible for loading and preprocessing speech command data
    for training, validation, or testing.

    Attributes:
        word_to_index: Mapping from word labels to numeric indices
        data: List of data samples with file paths and labels
        device: The torch device to use
    """

    VALID_SPLITS = ["training", "validation", "testing"]

    def __init__(
        self,
        dataset: SpeechCommandDataset,
        audio_processor: AudioProcessor,
        split: str = "training",
    ) -> None:
        """Initialize the data loader.

        Args:
            dataset: The speech command dataset
            audio_processor: The audio processor for feature extraction
            split: Dataset split to use ('training', 'validation', or 'testing')

        Raises:
            DatasetError: If the split is invalid
        """
        try:
            self.ap = audio_processor
            self.word_to_index = dataset.word_to_index

            if split not in self.VALID_SPLITS:
                raise DatasetError(
                    f"Invalid split: {split}. Must be one of {self.VALID_SPLITS}"
                )

            if split == "training":
                self.data = dataset.get_data("training")
                logger.info(f"Loading training data: {len(self.data)} samples")
            elif split == "validation":
                self.data = dataset.get_data("validation")
                logger.info(f"Loading validation data: {len(self.data)} samples")
            elif split == "testing":
                self.data = dataset.get_data("testing")
                logger.info(f"Loading testing data: {len(self.data)} samples")

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        except Exception as e:
            handle_error(
                e, DatasetError, f"Failed to initialize data loader for {split} split"
            )

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            Number of samples
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset.

        Args:
            index: Sample index

        Returns:
            Tuple of (audio features, label)

        Raises:
            DatasetError: If there's an error processing the sample
        """
        try:
            if index >= len(self.data):
                raise IndexError(
                    f"Index {index} out of bounds for dataset of size {len(self.data)}"
                )

            sample = self.data[index]
            filename = sample["file"]
            label = sample["label"]
            label_idx = self.word_to_index[label]

            # Extract features using audio processor
            signal = self.ap.transform(filename, label_idx)
            signal = torch.tensor(signal, dtype=torch.float32)
            signal = signal.unsqueeze(0)  # Add channel dimension
            return signal, label_idx

        except Exception as e:
            if isinstance(e, IndexError):
                raise
            handle_error(
                e,
                DatasetError,
                f"Error processing sample at index {index}",
                re_raise=True,
            )

    def get_class_mapping(self) -> Dict[int, str]:
        """Get mapping from class indices to labels.

        Returns:
            Dictionary mapping class indices to label strings
        """
        return {idx: word for word, idx in self.word_to_index.items()}
