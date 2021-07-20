import os
import torch
from torch.utils.data import Dataset
import python_speech_features as psf
import librosa
from . import input_data

# ignore warnings
import warnings

warnings.filterwarnings("ignore")


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
    print(len(kws_training_data))
    for i in range(len(kws_training_data)):
        sample = kws_training_data[i]
        print(i, sample[0], sample[1])
        break
