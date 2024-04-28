import torch
from loguru import logger
from torch.utils.data import Dataset
import torchaudio

from kws.datasets.speech_commands import SpeechCommandsDataset


class SpeechCommandsLoader(Dataset):
    def __init__(
        self,
        dataset: SpeechCommandsDataset,
        device: str,
        mode: str = "training",
        features: str = "mfcc",
    ) -> None:
        self.mode = mode
        self.device = device
        self.dataset = dataset
        if self.mode == "training":
            self.data = self.dataset.get_datafiles("training")
        else:
            self.data = self.dataset.get_datafiles("validation")
        super().__init__()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        filename = self.data[index]["file"]
        label = self.data[index]["label"]
        audio = self.dataset.audio_transform(filename, label)
        audio = torch.tensor(audio).to(self.device)
        return filename, label

    def signal2mfcc(
        self,
        signal: torch.Tensor,
        sample_rate: int,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mfcc: int = 13,
    ) -> torch.Tensor:
        mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": hop_length,
                "mel_scale": "htk",
                "n_mfcc": n_mfcc,
            },
        )
        return mfcc(signal)
