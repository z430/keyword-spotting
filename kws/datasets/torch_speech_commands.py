from pathlib import Path
from typing import List

from pydantic import BaseModel
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_tar, _load_waveform

from kws.utils.loader import load_yaml

DATA_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134MB
RANDOM_SEED = 59185
SILENCE_INDEX = 0
SILENCE_LABEL = "_silence_"
UNKNOWN_WORD_INDEX = 1
UNKNOWN_WORD_LABEL = "_unknown_"
BACKGROUND_NOISE_DIR_NAME = "_background_noise_"


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


class SpeechCommands(Dataset):
    def __init__(
        self, parameters_path: Path, dataset_path: Path, subset: str = "training"
    ) -> None:
        """Speech Commands Dataset

        Args:
            parameters_path (Path): parameters file path
            dataset_path (Path): dataset path
            subset (str, optional): dataset set. Defaults to "training".
        """
        self.parameters = Parameters(**load_yaml(parameters_path))
        self.dataset_path = dataset_path
