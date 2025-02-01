import argparse
from pathlib import Path

import numpy as np

from kws.datasets.speech_commands import DatasetConfig, SpeechCommandDataset


def train(opts):
    config = DatasetConfig()
    dataset = SpeechCommandDataset(config, Path("data/"))


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=str, default=None, help="Path to model weights"
    )
    return parser.parse_args()


if __name__ == "__main__":
    opts = parse_opt()
    train(opts)


# background_volume: 0.1
# background_frequency: 0.8
# time_shift_ms: 50.0
# sample_rate: 16000
# clip_duration_ms: 1000
# use_background_noise: True

# silence_percentage: 10.0
# unknown_percentage: 10.0
# testing_percentage: 10.0
# validation_percentage: 10.0
