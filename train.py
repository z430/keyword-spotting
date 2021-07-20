""" Training """
import logging
from os import device_encoding
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
import tqdm

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

    input_size = (next(iter(kws_training_data)))[0].shape

    net = models.LinearModel(input_size=input_size[1:], len_classes=len_classes)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    epochs = 3
    min_valid_loss = np.inf

    for e in range(epochs):
        train_loss = 0.0
        net.train()  # Optional when not using Model Specific layer
        for data, labels in tqdm.tqdm(train_dataloader):
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            data = data.float()
            labels = labels.to(device)
            optimizer.zero_grad()
            target = net(data)
            loss = criterion(target, labels)
            loss.backward()
            optimizer.step()
            train_loss = loss.item() * data.size(0)

        valid_loss = 0.0
        net.eval()  # Optional when not using Model Specific layer
        for data, labels in validation_dataloader:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()

            target = net(data)
            loss = criterion(target, labels)
            valid_loss = loss.item() * data.size(0)

        print(
            f"Epoch {e+1} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t Validation Loss: {valid_loss / len(validloader)}"
        )
        if min_valid_loss > valid_loss:
            print(
                f"Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model"
            )
            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save(net.state_dict(), "saved_model.pth")


if __name__ == "__main__":
    main()
