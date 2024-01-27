from pathlib import Path
import os

from loguru import logger
from torch.utils.data import DataLoader

from kws.libs.dataloader import SpeechCommandsLoader

ROOT_DIR = Path(__file__).parent
DATASET_PATH = ROOT_DIR / "data" / "speech-commands"


def main():
    train_loader = DataLoader(SpeechCommandsLoader(mode="training"))
    validation_loader = DataLoader(SpeechCommandsLoader(mode="validation"))

    for _ in range(10):
        for data, label in train_loader:
            logger.info(f"label: {label} filepath: {str(data)}")
            break


if __name__ == "__main__":
    main()
