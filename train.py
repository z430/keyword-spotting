from pathlib import Path

from loguru import logger

from kws.datasets.speech_commands import SpeechCommandsDataset

ROOT_DIR = Path(__file__).parent
DATASET_PATH = ROOT_DIR / "data" / "speech-commands"


def main():
    dataset = SpeechCommandsDataset(
        ROOT_DIR / "configs" / "speech_commands_parameters.yaml", DATASET_PATH
    )
    logger.info(dataset.parameters.time_shift)
    logger.info(dataset.parameters.desired_samples)


if __name__ == "__main__":
    main()

