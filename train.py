import argparse
from pathlib import Path


from kws.datasets.speech_commands import SpeechCommandsDataset
from kws.libs.dataloader import SpeechCommandsLoader, to_mfcc
from kws.settings import CONFIG_SPEECHCOMMANDS_DATASET_PATH, CONFIG_SPEECHCOMMANDS_PATH

FILE = Path(__file__).resolve()


def train(device, train_loader, validation_loader):
    # ...existing code...
    signal = np.ones(16000)
    features_shape = to_mfcc(
        signal, winlen=train_loader.frame_length, winstep=train_loader.frame_step
    ).shape
    # Add training loop and validation logic here
    # ...existing code...


def main():
    saved_weights_dir = ROOT_DIR / "saved_weights"
    saved_weights_dir.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = SpeechCommandsDataset(
        CONFIG_SPEECHCOMMANDS_PATH, CONFIG_SPEECHCOMMANDS_DATASET_PATH
    )

    train_loader = SpeechCommandsLoader(dataset, device, mode="training")
    validation_loader = SpeechCommandsLoader(dataset, device, mode="validation")

    train(device, train_loader, validation_loader)


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=str, default=None, help="Path to model weights"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
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
