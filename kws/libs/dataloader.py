from torch.utils.data import Dataset
from loguru import logger

from kws.datasets.speech_commands import SpeechCommandsDataset
from kws.settings import CONFIG_SPEECHCOMMANDS_DATASET_PATH, CONFIG_SPEECHCOMMANDS_PATH


class SpeechCommandsLoader(Dataset):
    def __init__(self, mode: str = "training") -> None:
        self.dataset = SpeechCommandsDataset(
            CONFIG_SPEECHCOMMANDS_PATH, CONFIG_SPEECHCOMMANDS_DATASET_PATH
        )
        self.mode = mode
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
        logger.info(audio.shape)
        return filename, label
