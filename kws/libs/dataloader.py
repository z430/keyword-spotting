import python_speech_features as psf
import torch
from loguru import logger
from torch.utils.data import Dataset

from kws.datasets.speech_commands import SpeechCommandsDataset


class SpeechCommandsLoader(Dataset):
    def __init__(
        self,
        dataset: SpeechCommandsDataset,
        device: str,
        mode: str = "training",
    ) -> None:
        self.mode = mode
        self.device = device
        self.dataset = dataset
        self.frame_length = 40 / 1000
        self.frame_step = 40 / 1000
        logger.info(f"Frame length: {self.frame_length}, Frame step: {self.frame_step}")
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
        signal = to_mfcc(audio, winlen=self.frame_length, winstep=self.frame_step)
        signal = torch.tensor(signal).to(self.device)
        signal = signal.view(-1)
        return signal, label


def to_mfcc(
    signal,
    samplerate=16000,
    numcep=10,
    nfft=512,
    winlen=0.025,
    winstep=0.01,
    nfilt=26,
):
    nfft = max(512, int(winlen * samplerate))
    return psf.mfcc(
        signal,
        samplerate=samplerate,
        numcep=numcep,
        nfft=nfft,
        winlen=winlen,
        winstep=winstep,
        nfilt=nfilt,
    )
