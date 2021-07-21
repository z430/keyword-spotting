from logging import root
from typing import Union, List, Dict, Optional
import os
import torch
from torch.utils.data import Dataset, DataLoader, dataloader
import python_speech_features as psf
import pytorch_lightning as pl

from . import input_data

# ignore warnings
import warnings

warnings.filterwarnings("ignore")


class LitKeywordSpotting(pl.LightningDataModule):
    def __init__(self, root_dir, wanted_words, batch_size):
        super().__init__()
        self.train_data = KeywordSpottingDataset(
            root_dir, wanted_words=wanted_words, mode="training"
        )
        self.validation_data = KeywordSpottingDataset(
            root_dir, wanted_words=wanted_words, mode="validation"
        )
        self.test_data = KeywordSpottingDataset(
            root_dir, wanted_words=wanted_words, mode="testing"
        )

        self.batch_size = batch_size
        self.total_clases = len(self.train_data.classes)
        self.input_size = (next(iter(self.train_data)))[0].shape

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.validation_data, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, batch_size=32)


class KeywordSpottingDataset(Dataset):
    def __init__(self, root_dir, wanted_words=None, transform=None, mode="training"):
        self.root_dir = root_dir
        self.transform = transform
        self.wanted_words = wanted_words
        self.data = input_data.GetData(
            wanted_words=self.wanted_words, data_dir=self.root_dir
        )
        self.mode = mode
        if self.mode not in ["training", "validation", "testing"]:
            raise ValueError("mode must be 'training', 'validation', 'testing'")

        self.datafiles = self.data.get_datafiles(self.mode)
        self.classes = self.data.words_list

        # config

    def __len__(self):
        return len(self.datafiles)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        datafile = self.datafiles[index]
        label = datafile["label"]
        label = self.data.word_to_index[label]
        filename = datafile["file"]
        waveform = self.data.audio_transform(filename, label)
        speech_feature = psf.mfcc(waveform, 16000)
        speech_feature = torch.from_numpy(speech_feature)
        speech_feature = speech_feature.unsqueeze(0).float()

        return speech_feature, label


if __name__ == "__main__":
    datadir = "../../data/train"
    wanted_words = [
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
    kws_training_data = KeywordSpottingDataset(
        root_dir=datadir, wanted_words=wanted_words
    )

    print(iter(next(kws_training_data)))
    print(len(kws_training_data))
    for i in range(len(kws_training_data)):
        sample = kws_training_data[i]
        print(i, sample[0], sample[1])
        break
