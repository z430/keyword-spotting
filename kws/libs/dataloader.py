from typing import Any, Dict, List

import librosa
import numpy as np
import python_speech_features as psf
import torch
from loguru import logger
from torch.utils.data import Dataset

from kws.datasets.speech_commands import SpeechCommandDataset
from kws.libs.signal_handler import AudioProcessor


class SpeechCommandsLoader(Dataset):
    def __init__(
        self,
        dataset: SpeechCommandDataset,
        audio_processor: AudioProcessor,
        split: str = "training",
    ) -> None:
        self.ap = audio_processor

        if split == "training":
            self.data = dataset.get_data("training")
        elif split == "validation":
            self.data = dataset.get_data("validation")
        elif split == "test":
            self.data = dataset.get_data("testing")
        else:
            raise ValueError(f"Invalid split: {split}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        filename = self.data[index]["file"]
        label = self.data[index]["label"]

        signal = self.ap.transform(filename, label)
        signal = torch.tensor(signal).to(self.device)
        signal = signal.view(-1)
        return signal, label
