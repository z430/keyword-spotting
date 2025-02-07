import argparse
from pathlib import Path

import numpy as np

from kws.datasets.speech_commands import DatasetConfig, SpeechCommandDataset
from kws.libs.signal_handler import AudioConfig, AudioProcessor
from kws.libs.dataloader import SpeechCommandsLoader


def train(opts):
    config = DatasetConfig()
    dataset = SpeechCommandDataset(config, Path("data/"))

    # audio processor
    audio_config = AudioConfig()
    audio_processor = AudioProcessor(dataset.root_dir, audio_config)

    # dataloader
    train_loader = SpeechCommandsLoader(dataset, audio_processor, "training")
    val_loader = SpeechCommandsLoader(dataset, audio_processor, "validation")

    for i, (signal, label) in enumerate(train_loader):  # type: ignore
        print(signal.shape, label)
        if i == 10:
            break


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=str, default=None, help="Path to model weights"
    )
    return parser.parse_args()


if __name__ == "__main__":
    opts = parse_opt()
    train(opts)
