import argparse
from pathlib import Path

from loguru import logger
from torch.utils.data import DataLoader

from kws.datasets.speech_commands import DatasetConfig, SpeechCommandDataset
from kws.libs.dataloader import SpeechCommandsLoader
from kws.libs.signal_handler import AudioConfig, AudioProcessor


def train(opts):
    config = DatasetConfig()
    dataset = SpeechCommandDataset(config, Path("data/"))

    # audio processor
    audio_config = AudioConfig()
    audio_processor = AudioProcessor(dataset.root_dir, audio_config)

    # dataloader
    train_loader = DataLoader(
        SpeechCommandsLoader(dataset, audio_processor, "training"),
        batch_size=32,
        shuffle=True,
    )
    val_loader = DataLoader(
        SpeechCommandsLoader(dataset, audio_processor, "validation"),
        batch_size=32,
        shuffle=True,
    )

    # take 1 sample from train_loader
    sample = next(iter(train_loader))
    input_shape = sample[0].shape

    logger.info(f"Training with input shape: {input_shape}")
    print(sample[0].shape)

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
