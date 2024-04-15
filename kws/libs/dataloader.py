import torch
from loguru import logger
from torch.utils.data import Dataset

from kws.datasets.speech_commands import SpeechCommandsDataset


class SpeechCommandsLoader(Dataset):
    def __init__(
        self, dataset: SpeechCommandsDataset, device: str, mode: str = "training"
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
        logger.info(audio.shape)
        return filename, label
