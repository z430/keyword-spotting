""" Training """
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from models import models
from utils import dataset

# global parameter
log = logging.getLogger(__name__)


def main():
    datadir = "../data/train"
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

    # get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using {device} device")

    kws_training_data = dataset.KeywordSpottingDataset(
        root_dir=datadir, wanted_words=wanted_words, mode="training"
    )
    kws_validation_data = dataset.KeywordSpottingDataset(
        root_dir=datadir, wanted_words=wanted_words, mode="validation"
    )

    train_dataloader = DataLoader(
        kws_training_data, batch_size=2, shuffle=True, pin_memory=True
    )

    validation_dataloader = DataLoader(
        kws_validation_data, batch_size=2, shuffle=True, pin_memory=True
    )

    # define the model
    len_classes = len(kws_training_data.classes)
    log.info(f"len classes -> {len_classes}")

    # check dataloader
    for i, (data, label) in enumerate(train_dataloader):
        print(i, data.shape, label.shape)
    # net = models.LinearModel(input_size=[0, 0], len_classes=len_classes)


if __name__ == "__main__":
    main()
