import os
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader

from kws.libs.dataloader import SpeechCommandsLoader

ROOT_DIR = Path(__file__).parent
DATASET_PATH = ROOT_DIR / "data" / "speech-commands"


def main():
    # check if there is cuda available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # set the seed for reproducibility
    torch.manual_seed(0)

    # set hyperparameters
    batch_size = 32
    learning_rate = 0.001
    epochs = 10

    # define the model
    model = torch.nn.Sequential(
        torch.nn.Linear(16000, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 12),
    )

    # define optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # create the dataloaders
    train_loader = DataLoader(SpeechCommandsLoader(mode="training"))
    validation_loader = DataLoader(SpeechCommandsLoader(mode="validation"))

    for _ in range(10):
        for data, label in train_loader:
            logger.info(f"label: {label} filepath: {str(data)}")
            break


if __name__ == "__main__":
    main()
